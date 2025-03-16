import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='XRFDataset')
    parser.add_argument('--data_dir', type=str, default='path/to/load', help='Dataset Path')
    parser.add_argument('--class_num', type=int, default=7, help='The Number of Classes')
    parser.add_argument('--epoch', type=int, default=100, help='The Number of Epoch[default: 100]')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate [default: 0.001]') #original: 0.001
    parser.add_argument('--model_num', type=int, default=1, help='The Number of Models for Mutual Learning')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch Size') #original: 64, cannot be 1!!!!
    parser.add_argument("--local_rank", type=int, default=1,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument('--use_noisy', action='store_true', default=True, 
                        help='Use noisy data for training (default: True)')

    args = parser.parse_args()

    return args
