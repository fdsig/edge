import wandb
import torch
import torchvision
import gc 
import platform

from config import args

if not args.inference:
    from ava import get_all, data_samplers, data_transforms, ava_data_reflect
else:
    from inference import get_all, data_samplers, data_transforms, ava_data_reflect

from download import get_dataset
if args.inference:
    
    print('evealuating model')
    from inference import deep_eval
else:
    import ava


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'    
    if args.download:
        assert args.dataset in ['ava']
        get_data = get_dataset(
            fid='crushed.zip', url='http://desigley.space/ava/crushed.zip')
        get_data.get_zip()
        get_data.unzip(out_dir='images')
    elif args.unzip_only:
        get_data = get_dataset(
            fid='../crushed.zip', url=None)
        get_data.unzip(out_dir='../images')
    

    df, y_g_dict, data, neg, pos = get_all(subset=args.subset)
    reflect_transforms = data_transforms(size=224)
    # data_loader = data_samplers(
    #     data=data, 
    #     reflect_transforms=reflect_transforms,
    #     ava_data_reflect=ava_data_reflect,
    #     batch_size=args.batch_size)
    if args.inference:

        wandb.login()
        print(f'running in inference mode{"8"*30}')
        wb_tags = ['inference', platform.system(), platform.system(),
                   platform.release()]
        if  args.entity and args.project and args.tags:
            run = wandb.init(entity=args.entity, project=args.project, tags=args.tags)
        elif args.entity and args.project:
            run = wandb.init(entity=args.entity, project=args.project)
        elif args.d.exists():
            with args.d.open('r') as hndl:
                for default in hndl.readlines():
                    default = default.split('=')
                    if len(default)==2:
                        arg_,param = default
                        arg_= arg_.strip('\n').strip(' ')
                        param = param.strip('\n').strip(' ')
                        args.arg_=param
            run = wandb.init(entity=args.entity, project=args.project)
        else:
            run = wandb.init()      
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512,2)
        loaded = torch.load('models/resnet_18',
                            map_location=torch.device(args.device))
        model.load_state_dict(loaded['model_state_dict'])
        run.watch(model)
        data_load_dict = data_samplers(data,ava_data_reflect,reflect_transforms,batch_size=128)
        evaluation = deep_eval(model, data_load_dict=data_load_dict)
        print('logging wandb table')
        run.finish()
    else:
        device = 'cuda'
        data_load_dict = data_samplers(data, ava_data_reflect, batch_size=128)
        torch.clear_autocast_cache()
        model = torch.nn.Conv2d(2, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        model.to(device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        did = Path('drive/MyDrive/0.AVA/results/training_eq_conditions/')
        if not did.exists():
            did.mkdir()
        res_did = Path('drive/MyDrive/0.AVA/results/')
        trained = [sub_dir.name for dir in did.iterdir()
                   for sub_dir in dir.iterdir()]
        avail_pretrained_models = timm.list_models(pretrained=True)
        nets = [net for net in avail_pretrained_models if 'mobilevit' in net and 'tf' not in net and 'rw' not in net]
        # set to multiple to train whole stack of models under equeal conditions
        nets = [nets[-1]]
        nets
        still_to_train = {net: {'location': did/net, 'epochs': 10}
                          for net in nets}
        mods = ava.loader(still_to_train)
