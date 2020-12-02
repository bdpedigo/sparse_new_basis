from pathlib import Path
import pandas as pd
import time

DATA_DIR = Path("sparse_new_basis/data/BP_Barabasi_Share/ScRNAData")


def load_scRNAseq(fillna=True, print_time=True):
    # sequencing data
    sequencing_loc = DATA_DIR / "Celegans_ScRNA_OnlyLabeledNeurons.csv"
    currtime = time.time()
    sequencing_df = pd.read_csv(sequencing_loc, skiprows=[1])
    
    # this appears to be faster than fillna during pivot using .pivot_table()
    sequencing_df = sequencing_df.pivot(
        index="neurons", columns="genes", values="Count"
    )
    if print_time:
        print(f"{time.time() - currtime:.3f} elapsed to load sequencing data.")
    if fillna:
        currtime = time.time()
        sequencing_df = sequencing_df.fillna(0)
        if print_time:
            print(f"{time.time() - currtime:.3f} elapsed to fillna.")

    # TODO drop the background RNA from table S2 in the paper

    # info about the genes themselves
    gene_loc = DATA_DIR / "GSE136049_gene_annotations.csv"
    currtime = time.time()
    gene_df = pd.read_csv(gene_loc)
    gene_df["genes"] = range(1, len(gene_df) + 1)
    gene_df = gene_df.set_index("genes")
    gene_df = gene_df.loc[sequencing_df.columns]  # some gene #s werent used
    sequencing_df.rename(columns=gene_df["gene_symbol"], inplace=True)
    if print_time:
        print(f"{time.time() - currtime:.3f} elapsed to load gene metadata.")

    # annotations for the individual genes
    annotation_genes = pd.read_csv(DATA_DIR / "annotation_genes.csv")
    currtime = time.time()
    nt_annotation_genes = annotation_genes.melt(
        id_vars=["neuron_class", "neuron_type"],
        value_vars=[f"nt_gene_{i}" for i in range(3)],
        value_name="gene",
    ).dropna(axis=0)
    nt_annotation_genes = nt_annotation_genes.drop("variable", axis=1)
    nt_annotation_genes["gene_type"] = "neurotransmitter"

    other_annotation_genes = annotation_genes.melt(
        id_vars=["neuron_class", "neuron_type"],
        value_vars=[f"gene_{i}" for i in range(12)],
        value_name="gene",
    ).dropna(axis=0)
    other_annotation_genes = other_annotation_genes.drop("variable", axis=1)
    other_annotation_genes["gene_type"] = "other"

    annotation_df = pd.concat((nt_annotation_genes, other_annotation_genes), axis=0)
    if print_time:
        print(f"{time.time() - currtime:.3f} elapsed to load annotation genes.")

    # metadata for each neuron in the gene expression data
    class_map_loc = DATA_DIR / "Labels2_CElegansScRNA_onlyLabeledNeurons.csv"
    neuron_df = pd.read_csv(class_map_loc)
    neuron_df = neuron_df.set_index("OldIndices")
    neuron_df.index.name = "neuron"

    currtime = time.time()
    assert (neuron_df.index.values == sequencing_df.index.values).all()
    neuron_df = neuron_df.reset_index()
    # sequencing_df = pd.concat((neuron_df, sequencing_df), axis=1)
    # sequencing_df.index.name = "neuron"
    # sequencing_df = sequencing_df.reset_index().set_index(
    #     ["neuron", "Neuron_type", "Experiment", "Barcode", "CellTypeIndex"]
    # )
    sequencing_df.index = pd.MultiIndex.from_frame(neuron_df)

    if print_time:
        print(f"{time.time() - currtime:.3f} elapsed for concatenation.")

    return sequencing_df, annotation_df
