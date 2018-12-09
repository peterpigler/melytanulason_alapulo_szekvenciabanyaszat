import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def validation_byrfaults(results_val1, results_val2, overall_results, filename):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    bp = ax.boxplot(results_val1*100)
    plt.ylabel('Corr. Pred. Rate (%)')
    plt.title('Pred. Seqs with good event in them',Fontsize=8)

    #ax.set_xticklabels(VnEVENT)
    ax.set_xticklabels([])
    plt.ylim((80,100))

    ax = fig.add_subplot(212)
    plt.title('Percent of correctly predicted events in seq.',Fontsize=8)
    bp = ax.boxplot(results_val2*100)
    plt.ylabel('Corr. Pred. Rate (%)')
    plt.xlabel('# of Fault')
    #plt.ylim((50,87))

    plt.savefig(filename, dpi=300)
    plt.show()


def validation_overall(overall_results, filename):

    overall_results = np.array(overall_results)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(overall_results*100)
    plt.ylabel('Corr. Pred. Rate (%)')
    plt.title('Pred. Seqs with good event in them',Fontsize=8)
    #ax.set_xticklabels(VnEVENT)
    ax.set_xticklabels(['val1', 'val2'])
    #plt.ylim((80,100))

    plt.savefig(filename, dpi=300)
    plt.show()


def vals_to_excel(results_val1, results_val2, overall_results, filename):

    results_all1 = np.insert(results_val1, 0, overall_results[:, 0],  axis=1)
    results_all2 = np.insert(results_val2, 0, overall_results[:, 1],  axis=1)

    adR1 = pd.DataFrame(results_all1)
    adR2 = pd.DataFrame(results_all2)
    writer = pd.ExcelWriter(filename)
    adR1.to_excel(writer,'val1')
    adR2.to_excel(writer,'val2')
    writer.save()


def validation_byfaults(results_val1, results_val2, overall_results, filename):

    results_all1 = np.insert(results_val1, 0, overall_results[:, 0],  axis=1)
    results_all2 = np.insert(results_val2, 0, overall_results[:, 1],  axis=1)

    results_all1 = np.delete(results_all1, 8, 1)
    results_all2 = np.delete(results_all2, 8, 1)
    xlabels = ['Overall', 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

    fig = plt.figure()
    ax = fig.add_subplot(211)
    bp = ax.boxplot(results_all1*100)
    plt.ylabel('$Val_1$ (%)')
    #plt.title('Pred. Seqs with good event in them',Fontsize=8)
    # #ax.set_xticklabels(VnEVENT)
    # ax.set_xticklabels([])
    # #plt.ylim((80,100))

    ax = fig.add_subplot(212)
    #plt.title('Percent of correctly predicted events in seq.',Fontsize=8)
    # bp = ax.boxplot(results_all2*100)
    # plt.ylabel('$Val_2$ (%)')

    ax.set_xticklabels(xlabels)
    plt.xlabel('# of Fault')
    #plt.ylim((50,87))

    plt.savefig(filename, dpi=300)
    plt.show()