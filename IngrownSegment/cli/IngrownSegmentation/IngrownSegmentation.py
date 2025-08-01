import os
import sys
from ctk_cli import CLIArgumentParser
import girder_client
sys.path.append('..')
from IngrownSegMain import run_main


DEFAULT_VALS = {
    "input_parameters":{
        "phase":"test",
        "type":"single",                
        "model_details":{				
            "architecture":"Unet++",
            "encoder":"resnet50",
            "encoder_weights":"imagenet",
            "dropout_rate":0.4,
            "active":"sigmoid",
            "target_type":"nonbinary",
            "in_channels":3,
            "ann_classes":"background,ingrowth"
        },
        "preprocessing":{
            "image_size":"512,512,3",
            "mask_size":"512,512,1",
            "color_transform":"None"
        }
    }
}
    

def main(args):

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    # Finding the id for the current WSI (input_image)
    file_id = args.input_file
    file_info = gc.get(f'/file/{file_id}')
    item_id = file_info['itemId']

    item_info = gc.get(f'/item/{item_id}')

    file_name = file_info['name']
    folder_id = item_info["folderId"]
    print(f'Running on: {file_name}')
    print(f'folder ID: {folder_id}')

    mounted_path = '{}/{}'.format('/mnt/girder_worker', os.listdir('/mnt/girder_worker')[0])
    file_path = '{}/{}'.format(mounted_path,file_name)
    gc.downloadFile(file_id, file_path)

    print(f'This is slide path: {file_path}')

    print('new version')
    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(args.input_file))

    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)

    for d in DEFAULT_VALS:
        if d not in list(vars(args).keys()):
            setattr(args,d,DEFAULT_VALS[d])

    setattr(args,'item_id', item_id)
    setattr(args,'file', file_path)
    setattr(args,'gc', gc)
    setattr(args,'folder_id', folder_id)

    print(vars(args))
    for d in vars(args):
        print(f'argument: {d}, value: {getattr(args,d)}')

    run_main(args)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())