import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS, LPIPSWithStyle
from taming.modules.losses.kl_loss import kl_loss
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_rec, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_rec = torch.mean(F.relu(1. + logits_rec))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    # d_loss = loss_real * .5 + (loss_rec + loss_fake) * .25
    d_loss = loss_real * .5 + loss_fake * .5
    return d_loss, loss_real, loss_rec, loss_fake


def vanilla_d_loss(logits_real, logits_rec, logits_fake):
    loss_real = torch.mean(torch.nn.functional.softplus(-logits_real))
    loss_rec = torch.mean(torch.nn.functional.softplus(logits_rec))
    loss_fake = torch.mean(torch.nn.functional.softplus(logits_fake))
    d_loss = loss_real * .5 + (loss_rec + loss_fake) * .25
    return d_loss, loss_real, loss_rec, loss_fake


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, kl_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, style_weight=0., use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        # if self.style_weight > 0:
        #     self.perceptual_loss = LPIPSWithStyle().eval().requires_grad_(False)
        # else:
        #     self.perceptual_loss = LPIPS().eval().requires_grad_(False)
        self.perceptual_loss = LPIPSWithStyle().eval().requires_grad_(False)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.__hidden__ = nn.Linear(1, 1, bias=False)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(self, codebook_loss, latent_var, latent_mean, inputs, reconstructions, fake, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if fake is not None:
            fake = fake.contiguous()

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

        if optimizer_idx == 0:
            # generator update
            rec_loss = torch.abs(inputs - reconstructions).mean()
            nll_loss = self.pixel_weight * rec_loss

            # if self.style_weight > 0:
            #     p_loss, s_loss = self.perceptual_loss(inputs, reconstructions)
            #     nll_loss += self.perceptual_weight * (p_loss + self.style_weight * s_loss)
            # else:
            #     p_loss = self.perceptual_loss(inputs, reconstructions)
            #     nll_loss += self.perceptual_weight * p_loss
            p_loss, s_loss = self.perceptual_loss(inputs, reconstructions)
            nll_loss += self.perceptual_weight * (p_loss + self.style_weight * s_loss)

            latent_kl_loss = kl_loss(latent_var / 0.558, latent_mean - 0.0009)

            loss = nll_loss + self.codebook_weight * codebook_loss + self.kl_weight * latent_kl_loss

            # the GAN part
            if disc_factor > 0:
                if cond is None:
                    assert not self.disc_conditional
                    logits_rec = self.discriminator(reconstructions)
                    logits_fake = self.discriminator(fake) \
                        if fake is not None else torch.tensor(0.).to(logits_rec.device)
                else:
                    assert self.disc_conditional
                    logits_rec = self.discriminator(torch.cat((reconstructions, cond), dim=1))
                    logits_fake = self.discriminator(torch.cat((fake, cond), dim=1)) \
                        if fake is not None else torch.tensor(0.).to(logits_rec.device)
                g_rec_loss = -torch.mean(logits_rec)
                g_fake_loss = -torch.mean(logits_fake)

                # try:
                #     adv_weight = self.calculate_adaptive_weight(nll_loss, g_rec_loss, last_layer=last_layer)
                #     adv_fake_weight = self.calculate_adaptive_weight(g_rec_loss, g_fake_loss, last_layer=last_layer) \
                #         if fake is not None else torch.zeros(1)
                # except RuntimeError:
                #     assert not self.training
                #     adv_weight, adv_fake_weight = torch.zeros(2)
                #
                # loss += disc_factor * adv_weight * (g_rec_loss + adv_fake_weight * g_fake_loss)
                try:
                    adv_weight = self.calculate_adaptive_weight(nll_loss, g_fake_loss, last_layer=last_layer)
                    adv_fake_weight = torch.zeros(1)
                except RuntimeError:
                    assert not self.training
                    adv_weight, adv_fake_weight = torch.zeros(2)

                loss += disc_factor * adv_weight * g_fake_loss

            else:
                adv_weight, adv_fake_weight, g_rec_loss, g_fake_loss = torch.zeros(4)

            loss = loss.mean()
            log = {
                "{}_supervised/quant_loss".format(split): codebook_loss.detach(),
                "{}_supervised/nll_loss".format(split): nll_loss.detach(),
                "{}_supervised/rec_loss".format(split): rec_loss.detach(),
                "{}_supervised/p_loss".format(split): p_loss.detach(),
                "{}_supervised/kl_loss".format(split): latent_kl_loss.detach(),
                "{}_stat/quant_mean".format(split): latent_mean.detach(),
                "{}_stat/quant_std".format(split): latent_var.detach().sqrt(),
                "{}_total/total_loss".format(split): loss.clone().detach().mean(),
            }
            # if self.style_weight > 0:
            #     log.update({"{}_supervised/style_loss".format(split): s_loss.detach()})
            log.update({"{}_supervised/style_loss".format(split): s_loss.detach()})
            if fake is not None:
                log.update({
                    "{}_adversarial_weight/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}_adversarial_weight/adv_weight".format(split): adv_weight.detach(),
                    "{}_adversarial_weight/d_fake_weight".format(split): adv_fake_weight.detach(),
                    "{}_adversarial_G/g_rec_loss".format(split): g_rec_loss.detach(),
                    "{}_adversarial_G/g_fake_loss".format(split): g_fake_loss.detach(),
                })
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if disc_factor > 0:
                if cond is None:
                    logits_real = self.discriminator(inputs)
                    logits_rec = self.discriminator(reconstructions)
                    logits_fake = self.discriminator(fake) \
                        if fake is not None else torch.tensor(1.).to(logits_rec.device)
                else:
                    logits_real = self.discriminator(torch.cat((inputs, cond), dim=1))
                    logits_rec = self.discriminator(torch.cat((reconstructions, cond), dim=1))
                    logits_fake = self.discriminator(torch.cat((fake, cond), dim=1)) \
                        if fake is not None else torch.tensor(1.).to(logits_rec.device)

                d_loss, d_loss_real, d_loss_rec, d_loss_fake = self.disc_loss(logits_real, logits_rec, logits_fake)
                d_loss *= disc_factor

            else:
                d_loss = torch.zeros((1, 1)).to(reconstructions.device)
                d_loss = d_loss + 0 * self.__hidden__(d_loss)  # let d_loss have grad_fn when using mixing precision
                d_loss_real, d_loss_rec, d_loss_fake = torch.zeros(3)

            log = {
                "{}_adversarial_D/disc_loss".format(split): d_loss.clone().detach(),
                "{}_adversarial_D/disc_loss_real".format(split): d_loss_real.detach(),
                "{}_adversarial_D/disc_loss_fake".format(split): d_loss_fake.detach(),
                "{}_adversarial_D/disc_loss_rec".format(split): d_loss_rec.detach(),
            } if fake is not None else dict()
            return d_loss, log
