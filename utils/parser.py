import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="last-fm",
                        help="Choose a dataset:[last-fm,amazon-book,alibaba-fashion]")
    parser.add_argument("--user_number", type=int, default="-1", help="limit user size:[-1 means all user]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_user_size', type=int, default=10, help='batch size')
    parser.add_argument('--batch_user_size_test', type=int, default=3000, help='batch size')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--dim1', type=int, default=1, help='embedding size1')
    parser.add_argument('--k_att', type=float, default=1, help='attention coe')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 5, 10,50, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()


def parse_args1():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="last-fm",
                        # parser.add_argument("--dataset", nargs="?", default="alibaba-fashion",
                        help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument("--user_number", type=int, default="-1", help="limit user size:[-1 means all user]")
    parser.add_argument("--number_file", type=int, default=10, help="limit user size:[-1 means all user]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=3000, help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--batch_user_size', type=int, default=10, help='batch size')
    parser.add_argument('--batch_user_size_test', type=int, default=3000, help='batch size')
    parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=32768, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=4096, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=512, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_pre', type=int, default=3, help='number of different prefer situatino')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--sim_regularity', type=float, default=1e-3, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.2, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.3, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=3, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='cosine', help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()


def parse_args2():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="last-fm",
                        # parser.add_argument("--dataset", nargs="?", default="alibaba-fashion",
                        help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument("--user_number", type=int, default="-1", help="limit user size:[-1 means all user]")
    parser.add_argument("--number_file", type=int, default=10, help="limit user size:[-1 means all user]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=2000, help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--batch_user_size', type=int, default=10, help='batch size')
    parser.add_argument('--batch_user_size_test', type=int, default=3000, help='batch size')
    parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=32768, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=4096, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=512, help='batch size')
    # parser.add_argument('--test_batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_pre', type=int, default=3, help='number of different prefer situatino')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--sim_regularity', type=float, default=1e-3, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.2, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.3, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=4, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='cosine', help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()
