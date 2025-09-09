import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="scene15", help='Dataset to use.')
    parser.add_argument("--layer_num", type=int, default=2, help="Number of network layer.")
    parser.add_argument('--rep_num', type=int, default=5, help='Number of rep.')
    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--k', type=int, default=10, help='k of kneighbors_graph.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nhid', type=int, default=16, help='the dimension of hidden layer')

    # for early stop
    parser.add_argument("--early-stop", type=bool, default=False, help="If early stop")
    parser.add_argument("--patience", type=int, default=20, help="Patence for early stop")

    parser.add_argument('--batch_norm', type=int, default=1, choices=[0, 1], help='whether to use batch norm')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='train_ratio')
    # parser.add_argument('--valid_ratio', type=float, default=0, help='valid_ratio')
    # parser.add_argument('--test_ratio', type=float, default=0.9, help='test_ratio')
    parser.add_argument('--data_split_mode', type=str, default='Ratio', help='data_split_mode, [Ratio, Num]')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--res_path', type=str, default="./results/GCN/", help='Dataset to use.')
    parser.add_argument('--device', type=str, default="0", help='gpu')

    args = parser.parse_args()

    return args
