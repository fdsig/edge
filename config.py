import argparse

parser = argparse.ArgumentParser(
    description='Training Binary Classifiers on AVA dataset for Edge computing/mobile device deployment')


parser.add_argument('--batch_size', default=None,
                    type=bool, help='batch_size')
parser.add_argument('--timm_model', default='mobile_vit_xxs',
                    type=str, help='modle_selected_from_timm_repository')
parser.add_argument('--n_workers', default='cpu', type=str, help='inference ')
parser.add_argument('--pin_memory', default=True, type=bool,
                    help='Handling of device memory pin == faster')
parser.add_argument('--inference', default=None, type=bool,
                    help='set computer brain state to infer or to train')
parser.add_argument('--device', default='cpu', type=str,
                    help='chose your device flavour this could ')
parser.add_argument('--model_location', default='models/mobilevit_xxs', type=str,
                    help='location relative to project directory where model is stored')
parser.add_argument('--download_from', default='', type=str,
                    help='location relative to project directory where model is stored')
parser.add_argument('--get_dataset', default=True, type=bool,
                    help='to get dataset or not to get dataset')
parser.add_argument('--download', default=False,
                    help='whether or not to download data for trining or for inference')

args = parser.parse_args()
