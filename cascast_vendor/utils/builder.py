# Stripped subset of the original CasCast utils/builder.py.
# Only ConfigBuilder.get_model() is kept — everything else (optimizer, lr
# scheduler, dataloader, megatron, timm) is removed because the course
# inference path calls get_model() exclusively.


class ConfigBuilder:
    def __init__(self, **params):
        self.model_params = params.get('model', {})
        self.logger = params.get('logger', None)

    def get_model(self, model_params=None):
        if model_params is None:
            model_params = self.model_params
        model_type = model_params.get('type', 'non_ar_model')
        params = model_params.get('params', {})
        if model_type == 'non_ar_model':
            from models.non_ar_model import non_ar_model
            return non_ar_model(self.logger, **params)
        elif model_type == 'autoencoder_kl_gan_model':
            from models.autoencoder_kl_gan_model import autoencoder_kl_gan_model
            return autoencoder_kl_gan_model(self.logger, **params)
        elif model_type == 'latent_diffusion_model_eval':
            from models.latent_diffusion_model_eval import latent_diffusion_model_eval
            return latent_diffusion_model_eval(self.logger, **params)
        else:
            raise NotImplementedError(f'Model type not supported in vendor: {model_type}')
