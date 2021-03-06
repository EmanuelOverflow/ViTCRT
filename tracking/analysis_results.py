import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.pytracking.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'otb'
"""vit_crt"""

trackers.extend(trackerlist(name='vit_crt', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='ViTCRT 320'))
trackers.extend(trackerlist(name='vit_crt', parameter_name='baseline_380', dataset_name=dataset_name,
                            run_ids=None, display_name='ViTCRT 380'))


dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
#dataset = get_dataset('otb')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
