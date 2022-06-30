import _init_paths
import argparse
from vit_crt import ViTCRT
import got10k.experiments as exps

parser = argparse.ArgumentParser(description='vit_cr tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--result_dir', default='results', type=str,
        help='config file')
parser.add_argument('--report_dir', default='reports', type=str,
        help='config file')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

if __name__ == '__main__':

    # setup tracker
    tracker = ViTCRT(param_name=args.config, dataset_name="OTB")
    dataset = args.dataset.lower()
    # run experiments on GOT-10k (validation subset)
    if dataset == "got10k":
        experiment = exps.ExperimentGOT10k(args.dataset_path, subset='test',
                                           result_dir=args.result_dir, report_dir=args.report_dir)
    if dataset == "got10k_val":
        experiment = exps.ExperimentGOT10k(args.dataset_path, subset='val',
                                           result_dir=args.result_dir, report_dir=args.report_dir)
    elif dataset == "otb2015":
        experiment = exps.ExperimentOTB(args.dataset_path, version=2015,
                                        result_dir=args.result_dir, report_dir=args.report_dir)
    elif dataset == "vot2019":
        experiment = exps.ExperimentVOT(args.dataset_path, version=2019,
                                        result_dir=args.result_dir, report_dir=args.report_dir)
    elif dataset == "lasot":
        experiment = exps.ExperimentLaSOT(args.dataset_path,
                                          result_dir=args.result_dir, report_dir=args.report_dir)
    elif dataset == 'trackingnet':
        experiment = exps.ExperimentTrackingNet(args.dataset_path,
                                                result_dir=args.result_dir, report_dir=args.report_dir)
    elif dataset == 'tc128':
        experiment = exps.ExperimentTColor128(args.dataset_path,
                                              result_dir=args.result_dir, report_dir=args.report_dir)
    elif dataset == 'uav':
        experiment = exps.ExperimentUAV123(args.dataset_path,
                                           result_dir=args.result_dir, report_dir=args.report_dir)
    elif dataset == 'nfs':
        experiment = exps.ExperimentNfS(args.dataset_path,
                                        result_dir=args.result_dir, report_dir=args.report_dir)

    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])
