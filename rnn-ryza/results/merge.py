import os
import pandas as pd

main_dir = '/Users/angelaoryza/Documents/TA/noisy-rnnids/rnnids-py/results/'

# w = pd.ExcelWriter(main_dir+'all_score.xlsx')

subdir = os.listdir(main_dir)
print(subdir)

with pd.ExcelWriter(main_dir + 'all_score_http-2.xlsx') as w:
    for dir in subdir:
        if dir.endswith('.pcap'):
            for file in os.listdir(dir):
                if file.endswith('.xlsx'):
                    print(file)
                    if "http" not in file:
                        continue
                    else:
                        df = pd.read_excel(f'./{dir}/{file}', header = None)
                        print(df)
                        new_sheet_name = os.path.basename(file).split('.xlsx')[0]
                        print(new_sheet_name)

                        df.to_excel(w, sheet_name = new_sheet_name , index = False)
        else:
            continue