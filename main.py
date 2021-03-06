from ast import arg
import wandb
import torch 
import timm 
import platform

from config import args
from ava import get_all,data_samplers, data_transforms, ava_data_reflect
from download import get_dataset
if args.inference:
    from inference import deep_eval
else:
    import ava

    

#hugging face api autonlp argugments (passed back to lower level in stack) 



if __name__=='__main__':
    wandb.login()
    if args.get_dataset:
        get_data = get_dataset(fid='crushed.zip',url = 'http://desigley.space/ava/crushed.zip')
        get_data.get_zip()
        get_data.unzip(out_dir='images')

    
    df,y_g_dict, data, neg, pos = get_all(subset=True)
    reflect_transforms = data_transforms(size=224)
    data_load_dict = data_samplers(data,ava_data_reflect,batch_size=args.batch_size)
    if args.inference:
        wb_tags = ['inference',platform.system(),platform.system(),platform.release()]
        wandb.init(entity='iaqa',project='small_is_beautiful',tags=wb_tags)
        model = timm.create_model('mobilevit_xxs')
        model.head.fc.out_features=2
        loaded = torch.load('models/mobilevit_xxs',map_location=torch.device(args.device))
        model.load_state_dict(loaded['model_state_dict'])
        wandb.watch(model)
        evaluation = deep_eval(model)
        print('logging wandb table')
        wandb.finish()
    else:
        device = 'cuda'
        data_load_dict = data_samplers(data,ava_data_reflect,batch_size=128)
        torch.clear_autocast_cache()
        model = torch.nn.Conv2d(2,64,kernel_size=(3,3), stride=(1,1),padding=(1,1))
        model.to(device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        did = Path('drive/MyDrive/0.AVA/results/training_eq_conditions/')
        if not did.exists():
            did.mkdir()
        res_did = Path('drive/MyDrive/0.AVA/results/')
        trained = [sub_dir.name for dir in did.iterdir() for sub_dir in dir.iterdir()]
        avail_pretrained_models = timm.list_models(pretrained=True)
        nets = [net for net in avail_pretrained_models if 'mobilevit' in net and 'tf' not in net and 'rw' not in net]
        # set to multiple to train whole stack of models under equeal conditions
        nets=[nets[-1]]
        nets
        still_to_train = {net:{'location':did/net,'epochs':10} for net in nets}
        mods = ava.loader(still_to_train)


    
   
    
    