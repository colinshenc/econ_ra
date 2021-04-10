#########################
#			            #
#	Oct 13, 2020	    #
# Created by Cheng Shen #
#			            #
#########################
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
import weighted

if __name__ == "__main__":
    file = pd.read_csv("RA_21_22.csv")
    # print(file.head())
    # print(file.columns)
    file['wealth_total'] = file.apply(lambda row: row['asset_total'] - row['debt_total'], axis=1)
    file['wealth_housing'] = file.apply(lambda row: row['asset_housing'] - row['debt_housing'], axis=1)
    file['wealth_non_housing'] = file.apply(lambda row: row['wealth_total']-row['wealth_housing'], axis=1)

    year_list = file['year'].unique()
    race_list = file['race'].unique()
    # print(file.columns)
    # print(file[:100][['wealth_total','age','weight']])
    # print(file[:5][['asset_total','weight']])
    # print(file[:5][['debt_total','weight']])

    ed_list = file['education'].unique()
    # a = file.loc[(file['wealth_total'] < 0) &(file['wealth_housing'] < 0)].sort_values('wealth_housing')
    # print(a[:300])
    run_q1=True
    run_q3=True
    if run_q1:
        '''Q1,Q2'''
        # print(file['race'].unique())
        # print(file['asset_total'].head())
        # print(file['debt_total'].head())
        # print(file['wealth_total'].head())

        year_group = file.groupby('year')

        year_education_dict_ = {}
        year_race_dict_ = {}
        # print(file['weight'].head())
        # for i in file['year'].unique():
        #     year_race_dict_[i] = year_group.get_group(i).groupby('race')
        #     year_education_dict_[i] = year_group.get_group(i).groupby('education')

        # year_race_dict = {}
        year_race_medians_dict = {}
        # year_education_dict = {}
        year_education_medians_dict = {}
        # year_list = file['year'].unique()
        # race_list = file['race'].unique()
        # ed_list = file['education'].unique()
        year_race_h_medians_dict = {}
        for i in year_list:
            i = int(i)
            # year = year_group.get_group(i)#.groupby('race')
            year = file.loc[file['year']==i]
            year_education_dict_[i] = year_group.get_group(i).groupby('education')
            # year_race_dict[i]=[]
            year_race_medians_dict[i] = []
            # year_education_dict[i]=[]
            year_education_medians_dict[i] = []
            year_race_h_medians_dict[i] = []
            for j in race_list:
                # print(i)
                # print(j)
                year_race = year.loc[year['race']==j].sort_values('wealth_total')
                # year_race = year_race_dict_[i].get_group(j).sort_values('wealth_total')
                year_race_h = year.loc[year['race']==j].sort_values('wealth_housing')
                # print('sorted \n{}'.format(year_race['wealth_total'].head()))
                # print('00000')
                # print(year_race_h['year'].unique)
                # print(year_race_h['race'].unique)


                temp = year_race['weight'].cumsum()
                temp_h = year_race_h['weight'].cumsum()

                half = year_race['weight'].sum() * 0.5
                # print('half2{}'.format(half))
                half_h = year_race_h['weight'].sum() / 2
                # print('wealth housing\n{}'.format(year_race_h['wealth_housing'][:1000]))
                # print('asset housing\n{}'.format(year_race_h['asset_housing'][:100]))
                # print('debt housing\n{}'.format(year_race_h['debt_housing'][:100]))

                # print('\ntemp\n{}..... \nhalf\n{}'.format(temp_h[:100], half_h[:100]))
                '''weighted median calc method 1, no interp'''

                weighted_median = year_race[temp >= half]['wealth_total'].iloc[0]
                weighted_median_h = year_race_h['wealth_housing'][temp_h >= half_h].iloc[0]
                # print('----> {}'.format(year_race_h['wealth_housing'][temp_h <= half_h][:100]))
                # print('\ntemp{}..... \nhalf{}'.format(temp,half))
                # print('\nmed {}'.format(weighted_median_h))
                '''weighted median calc method 2, interp'''
                # weighted_median = weighted.median(year_race['debt_total'], year_race['weight'])
                # weighted_median_h = weighted.median(year_race['wealth_housing'], year_race['weight'])

                year_race_medians_dict[i].append(weighted_median)
                # print('{},{},{}'.format(i,j,weighted_median))
                # print('\n\n\n')

                year_race_h_medians_dict[i].append(weighted_median_h)
                # print('median {}'.format(year_race_medians_dict[i][-1]))
                # print(year_race_dict[i][-1]['asset_total'].head())
                # print(year_race_dict[i][-1]['debt_total'].head())
                # print(year_race_dict[i][-1]['wealth_total'].head())
            for j in ed_list:
                # print(i)
                # print(j)
                # print(year_education_dict[i][-1]['year'].unique)
                # print(year_education_dict[i][-1]['education'].unique)
                year_education = year_education_dict_[i].get_group(j).sort_values('wealth_total',)
                # print('sorted \n{}'.format(year_education['wealth_total'].head()))
                temp = year_education['weight'].cumsum()
                half = year_education['weight'].sum() * 0.5
                # print('\ntemp\n{}..... \nhalf{}'.format(temp,half))

                weighted_median = year_education['wealth_total'][temp >= half].iloc[0]
                # print('\ntemp{}..... \nhalf{}'.format(temp,half))
                # print('\nmed {}'.format(weighted_median))
                year_education_medians_dict[i].append(weighted_median)
                # print('median {}'.format(year_education_medians_dict[i][-1]))
                # print(year_education_dict[i][-1]['wealth_total'].head())

                # print('\n\n\n')

        with open('./plot_data_.txt', 'a+') as f:
            f.write('\n------Q1 RACE DATA------\n')
            f.write(json.dumps(year_race_medians_dict))
            f.write('\n------Q1 EDUCATION DATA------\n')
            f.write(json.dumps(year_education_medians_dict))
            f.write('\n------Q2 DATA------\n')
            f.write(json.dumps(year_race_h_medians_dict))
            # f.write('\n')

        fig = plt.figure()

        # bins = list(year_list)
        colors=['g','c','y','r','m','b']
        # markers=['*','^','.']
        fig, ax = plt.subplots(figsize=(11,7))
        fig1, ax1 = plt.subplots(figsize=(11,7))
        fig2, ax2 = plt.subplots(figsize=(11,7))

        w = 0.6

        # ax.xaxis_date()
        # ax.autoscale(tight=True)
        # for i in range(len(race_list)):
        #     print(year_list)
        ax.bar(year_list-1.5*w,[year_race_medians_dict[_][0] for _ in year_list], w, alpha=0.65, color=colors[0], label=race_list[0], align='center')
        ax.bar(year_list-0.5*w,[year_race_medians_dict[_][1] for _ in year_list], w, alpha=0.65, color=colors[1], label=race_list[1], align='center')
        ax.bar(year_list+0.5*w,[year_race_medians_dict[_][2] for _ in year_list], w, alpha=0.65, color=colors[2], label=race_list[2], align='center')
        ax.bar(year_list+1.5*w,[year_race_medians_dict[_][3] for _ in year_list], w, alpha=0.65, color=colors[3], label=race_list[3], align='center')

        ax1.bar(year_list -  w, [year_education_medians_dict[_][0] for _ in year_list], w, alpha=0.65, color=colors[0],
               label=ed_list[0], align='center')
        ax1.bar(year_list, [year_education_medians_dict[_][1] for _ in year_list], w, alpha=0.65, color=colors[1],
               label=ed_list[1], align='center')
        ax1.bar(year_list + w, [year_education_medians_dict[_][2] for _ in year_list], w, alpha=0.65, color=colors[2],
               label=ed_list[2], align='center')

        ax2.bar(year_list - 0.5 * w, [year_race_h_medians_dict[_][0] for _ in year_list], w, alpha=0.65, color=colors[0],
               label=race_list[0], align='center')
        ax2.bar(year_list + 0.5 * w, [year_race_h_medians_dict[_][1] for _ in year_list], w, alpha=0.65, color=colors[1],
               label=race_list[1], align='center')
        # ax2.bar(year_list + 0.5 * w, [year_race_h_medians_dict[_][2] for _ in year_list], w, alpha=0.65, color=colors[2],
        #        label=race_list[2], align='center')
        # ax2.bar(year_list + 1.5 * w, [year_race_h_medians_dict[_][3] for _ in year_list], w, alpha=0.65, color=colors[3],
        #        label=race_list[3], align='center')


        ax.autoscale(tight=False)
        ax.set_title('Median total wealth by race groups')

        ax.set_xticks(year_list)
        ax.legend(loc='upper left')
        ax.set_ylabel('2016 U.S.$')
        ax.set_xlabel('Year')

        ax1.autoscale(tight=False)
        ax1.set_title('Median total wealth by education groups')
        ax1.set_xticks(year_list)
        # ax.set_yticks([5e+5 * k for k in range(9)])

        ax1.legend(loc='upper right')
        ax1.set_ylabel('2016 U.S.$')
        ax1.set_xlabel('Year')


        ax2.autoscale(tight=False)
        ax2.set_title('Median housing wealth by race groups(black and white)')

        ax2.set_xticks(year_list)
        ax2.legend(loc='upper right')
        ax2.set_ylabel('2016 U.S.$')
        ax2.set_xlabel('Year')

        fig.savefig('/ubc/cs/research/shield/projects/cshen001/econ_ra_task/{}.png'.format('plot_q1_race'), dpi=600)
        fig1.savefig('/ubc/cs/research/shield/projects/cshen001/econ_ra_task/{}.png'.format('plot_q1_education'), dpi=600)
        fig2.savefig('/ubc/cs/research/shield/projects/cshen001/econ_ra_task/{}.png'.format('plot_q2'), dpi=600)

    if run_q3:


        '''Q3'''
        'homeowners over 25'
        # print(type(file))
        above_25 = file.loc[(file['asset_housing']>0) & (file['age'] >= 25)].sort_values('age')

        black = above_25.loc[above_25['race']=='black']
        white = above_25.loc[above_25['race']=='white']

        black_ = black.groupby('year')
        white_ = white.groupby('year')

        bnh_wa = black_.apply(lambda x: pd.Series(np.average(x['wealth_non_housing'], weights=x['weight'], axis=0), ['wealth_non_housing'])).to_dict()
        # print(bnh_wa)
        # a = (black['wealth_non_housing'].loc[black['year']==2016]*black['weight'].loc[black['year']==2016]).sum()/(black['weight'].loc[black['year']==2016]).sum()
        # print(a)
        bh_wa = black_.apply(lambda x: pd.Series(np.average(x['wealth_housing'], weights=x['weight'], axis=0), ['wealth_housing'])).to_dict()
        # print(bh_wa)
        # b = (black['wealth_housing'].loc[black['year']==2016]*black['weight'].loc[black['year']==2016]).sum()/(black['weight'].loc[black['year']==2016]).sum()
        # print(b)

        wnh_wa = white_.apply(lambda x: pd.Series(np.average(x['wealth_non_housing'], weights=x['weight'], axis=0), ['wealth_non_housing'])).to_dict()
        # print(wnh_wa)
        # a = (white['wealth_non_housing'].loc[white['year'] == 2016] * white['weight'].loc[white['year'] == 2016]).sum() / (white['weight'].loc[white['year'] == 2016]).sum()
        # print(a)
        wh_wa = white_.apply(lambda x: pd.Series(np.average(x['wealth_housing'], weights=x['weight'], axis=0), ['wealth_housing'])).to_dict()
        # print(wh_wa)
        # b = (white['wealth_housing'].loc[white['year'] == 2016] * white['weight'].loc[white['year'] == 2016]).sum() / (white['weight'].loc[white['year'] == 2016]).sum()
        # print(b)




        diff_bh_2010 = bh_wa['wealth_housing'][2010] - bh_wa['wealth_housing'][2007]
        r_diff_bh_2010 = diff_bh_2010 / bh_wa['wealth_housing'][2007]
        # print(r_diff_bh_2010)
        # print(diff_bh_2010)

        diff_bh_2013 = bh_wa['wealth_housing'][2013]-bh_wa['wealth_housing'][2007]
        r_diff_bh_2013 = diff_bh_2013 / bh_wa['wealth_housing'][2007]
        # print(r_diff_bh_2013)
        # print(diff_bh_2013)

        diff_bnh_2010 = bnh_wa['wealth_non_housing'][2010] - bnh_wa['wealth_non_housing'][2007]
        r_diff_bnh_2010 = diff_bnh_2010 / bnh_wa['wealth_non_housing'][2007]
        # print(r_diff_bnh_2010)
        # print(diff_bnh_2010)
        diff_bnh_2013 = bnh_wa['wealth_non_housing'][2013] - bnh_wa['wealth_non_housing'][2007]
        r_diff_bnh_2013 = diff_bnh_2013 / bnh_wa['wealth_non_housing'][2007]
        # print(r_diff_bnh_2013)
        # print(diff_bnh_2013)

        diff_wh_2010 = wh_wa['wealth_housing'][2010] - wh_wa['wealth_housing'][2007]
        r_diff_wh_2010 = diff_wh_2010 / wh_wa['wealth_housing'][2007]
        # print(r_diff_wh_2010)
        # print(diff_wh_2010)

        diff_wh_2013 = wh_wa['wealth_housing'][2013] - wh_wa['wealth_housing'][2007]
        r_diff_wh_2013 = diff_wh_2013 / wh_wa['wealth_housing'][2007]
        # print(r_diff_wh_2013)
        # print(diff_wh_2013)

        diff_wnh_2010 = wnh_wa['wealth_non_housing'][2010] - wnh_wa['wealth_non_housing'][2007]
        r_diff_wnh_2010 = diff_wnh_2010 / wnh_wa['wealth_non_housing'][2007]
        # print(r_diff_wnh_2010)
        # print(diff_wnh_2010)

        diff_wnh_2013 = wnh_wa['wealth_non_housing'][2013] - wnh_wa['wealth_non_housing'][2007]
        r_diff_wnh_2013 = diff_wnh_2013 / wnh_wa['wealth_non_housing'][2007]
        # print(r_diff_wnh_2013)
        # print(diff_wnh_2013)




        fig = plt.figure()
        plt.plot(year_list, [bnh_wa['wealth_non_housing'][_] for _ in year_list], color='c', label='black non-housing wealth',linestyle=':', alpha=0.65)
        plt.plot(year_list, [bh_wa['wealth_housing'][_] for _ in year_list], color='c', label='black housing wealth', alpha=0.9)
        plt.plot(year_list, [wnh_wa['wealth_non_housing'][_] for _ in year_list], color='m', label='white non-housing wealth', linestyle=':', alpha=0.65)
        plt.plot(year_list, [wh_wa['wealth_housing'][_] for _ in year_list], color='m', label='white housing wealth', alpha=0.9)

        plt.xticks(year_list)
        # plt.xticklabels(feat_mult)
        plt.legend(loc="upper left",prop={'size':8})
        plt.xlabel('Year')
        plt.ylabel('U.S.$')
        plt.title('Mean housing, non-housing wealth by race')

        plt.savefig('/ubc/cs/research/shield/projects/cshen001/econ_ra_task/{}.png'.format('plot_q3'), dpi=600)



    with open('./plot_data_.txt', 'a+') as f:
        # file.write('\n------Q1 RACE DATA------\n')
        # file.write(json.dumps(year_race_medians_dict))
        # file.write('\n------Q2 EDUCATION DATA------\n')
        # file.write(json.dumps(year_education_medians_dict))
        # file.write('\n------Q2 DATA------\n')
        # file.write(json.dumps(year_race_h_medians_dict))
        f.write('\n------Q3 DATA------\n')
        f.write(json.dumps({'bnh_wa':bnh_wa,'bh_wa':bh_wa,'wnh_wa':wnh_wa, 'wh_wa':wh_wa}))
        f.write('\n')
        f.write(json.dumps({'diff_bh_2010':diff_bh_2010, 'diff_bh_2013':diff_bh_2013, 'diff_wh_2010':diff_wh_2010, 'diff_wh_2013':diff_wh_2013,
                               'diff_bnh_2010':diff_bnh_2010, 'diff_bnh_2013':diff_bnh_2013, 'diff_wnh_2010': diff_wnh_2010, 'diff_wnh_2013': diff_wnh_2013, }))

        f.write('\n\n\n')