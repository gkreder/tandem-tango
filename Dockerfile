FROM continuumio/miniconda3

RUN conda install -y -c conda-forge -c bioconda ipykernel lxml molmass matplotlib more-itertools cvxpy pandas seaborn snakemake ipdb tqdm
RUN pip install pyteomics xlsxwriter adjusttext prefect joblib openpyxl
