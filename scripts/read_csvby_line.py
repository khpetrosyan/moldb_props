from tqdm import tqdm

def read_csv_line_by_line(input_csv_path, output_csv_path, col_idx):
    results = []
    with open(input_csv_path, 'r') as f:
        for line in tqdm(f.readlines()):
            line_res = line.split("\t")[col_idx+1]
            results.append(line_res)
            
    with open(output_csv_path, "w") as f:
        f.write("\n".join(results))

read_csv_line_by_line('/home/khoren/moldb_props/data/BindingDB_All.tsv', '/home/khoren/all_smiles.csv', 0)
