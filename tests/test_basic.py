import numpy as np
import pandas as pd

import flowsom as fs


def test_package_has_version():
    assert fs.__version__


def test_fcs(FlowSOM_res):
    assert FlowSOM_res.get_cell_data().shape == (19225, 18)


def test_csv(fcs_path):
    ff = fs.io.read_csv(fcs_path)
    fs.FlowSOM(ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], n_clusters=10)


def test_FlowSOM_type(FlowSOM_res):
    assert isinstance(FlowSOM_res, fs.FlowSOM)


def test_FlowSOM_cluster_labels(FlowSOM_res):
    assert len(np.unique(FlowSOM_res.cluster_labels)) == 100


def test_FlowSOM_metacluster_labels(FlowSOM_res):
    assert len(np.unique(FlowSOM_res.metacluster_labels)) == 10


def test_plot_stars(FlowSOM_res, tmp_path):
    pl = fs.pl.plot_stars(
        FlowSOM_res,
        background_values=FlowSOM_res.get_cluster_data().obs["metaclustering"],
        view="MST",
        equal_node_size=False,
    )
    pl.savefig(tmp_path / "plotstars.pdf")


def test_plot_marker(FlowSOM_res, tmp_path):
    pl = fs.pl.plot_marker(
        FlowSOM_res,
        marker=["CD3"],
        background_values=FlowSOM_res.get_cluster_data().obs["metaclustering"],
        view="grid",
        equal_node_size=True,
        equal_background_size=True,
    )
    pl.savefig(tmp_path / "plotmarker.pdf")


def test_plot_pies(FlowSOM_res, gating_results, tmp_path):
    pl = fs.pl.plot_pies(
        FlowSOM_res,
        cell_types=gating_results,
        background_values=FlowSOM_res.get_cluster_data().obs["metaclustering"],
    )
    pl.savefig(tmp_path / "plotpies.pdf")


def test_plot_variable(FlowSOM_res, tmp_path):
    pl = fs.pl.plot_variable(
        FlowSOM_res,
        variable=FlowSOM_res.get_cluster_data().obs["metaclustering"],
        labels=FlowSOM_res.get_cluster_data().obs["metaclustering"],
        cmap=fs.pl.gg_color_hue(),
    )
    pl.savefig(tmp_path / "plotvariable.pdf")


def test_plot_2D_scatters(FlowSOM_res, tmp_path):
    pl = fs.pl.plot_2D_scatters(
        FlowSOM_res,
        channelpairs=[["CD3", "CD4"], ["CD19", "TCRb"]],
        clusters=[[1, 2], [3]],
        metaclusters=[[4], [5, 6]],
        density=False,
        centers=True,
    )
    pl.savefig(tmp_path / "plot2Dscatters.pdf")


def test_new_data(fcs):
    fsom = fs.FlowSOM(fcs[0:5000, :], cols_to_use=[8, 11, 13, 14, 15, 16, 17], n_clusters=10)
    fsom_new = fsom.new_data(fcs[1001:2000])
    assert fsom_new.get_cell_data().shape == (999, 18)


def test_flowsom_clustering(fcs):
    inp = fs.flowsom_clustering(fcs, cols_to_use=[8, 11, 13, 14, 15, 16, 17])
    assert "FlowSOM_clusters" in inp.obs.keys()
    assert "FlowSOM_metaclusters" in inp.obs.keys()
    assert "FlowSOM" in inp.uns.keys()


def test_aggregate_flowframes(ff_path):
    new_ff = fs.pp.aggregate_flowframes(
        [str(ff_path), str(ff_path)], c_total=5000, channels=[8, 11, 13, 14, 15, 16, 17]
    )
    assert new_ff.shape == (5000, 7)


def test_flowsom_subset(FlowSOM_res):
    fsom_subset = FlowSOM_res.subset(FlowSOM_res.get_cell_data().obs["metaclustering"] == 1)
    fs.pl.plot_stars(fsom_subset, background_values=fsom_subset.get_cluster_data().obs["metaclustering"])
    assert fsom_subset.get_cell_data().shape == (sum(FlowSOM_res.get_cell_data().obs["metaclustering"] == 1), 18)


def test_get_features(FlowSOM_res, ff_path):
    fs.tl.get_features(
        FlowSOM_res,
        [str(ff_path), str(ff_path)],
        level=["clusters", "metaclusters"],
        type=["counts", "MFIs", "percentages", "percentages_positive"],
        MFI=["CD3", "CD4"],
        positive_cutoffs={"CD3": 2, "PE-A": 2},
        filenames=None,
    )


def test_FlowSOMmary(FlowSOM_res, tmp_path):
    fs.pl.FlowSOMmary(FlowSOM_res, tmp_path / "FlowSOMmary.pdf")


def test_FlowSOM_class(FlowSOM_res):
    cell_data = FlowSOM_res.get_cell_data()
    uns_true_keys_cell = ["n_nodes", "n_metaclusters"]
    obs_true_keys_cell = ["clustering", "distance_to_bmu", "metaclustering"]
    var_true_keys_cell = ["pretty_colnames", "cols_used", "markers", "channels"]

    # make sure that at least the true keys are in the object
    uns_keys_cell = all(key in cell_data.uns.keys() for key in uns_true_keys_cell)
    obs_keys_cell = all(key in cell_data.obs.keys() for key in obs_true_keys_cell)
    var_keys_cell = all(key in cell_data.var.keys() for key in var_true_keys_cell)

    cluster_data = FlowSOM_res.get_cluster_data()
    uns_true_keys_cluster = ["outliers", "graph", "xdim", "ydim", "metacluster_MFIs"]
    obs_true_keys_cluster = ["percentages", "metaclustering"]
    obsm_true_keys_cluster = ["cv_values", "sd_values", "mad_values", "codes", "grid", "layout"]

    # make sure that at least the true keys are in the object
    uns_keys_cluster = all(key in cluster_data.uns.keys() for key in uns_true_keys_cluster)
    obs_keys_cluster = all(key in cluster_data.obs.keys() for key in obs_true_keys_cluster)
    obsm_keys_cluster = all(key in cluster_data.obsm.keys() for key in obsm_true_keys_cluster)
    assert all(
        [
            uns_keys_cell,
            obs_keys_cell,
            var_keys_cell,
            uns_keys_cluster,
            obs_keys_cluster,
            obsm_keys_cluster,
        ]
    )


def test_mfis():
    # create a DataFrame with 4 cells and 4 markers with values in non-ascending order
    markers = ["CD3", "CD4", "CD8", "CD19"]
    data = pd.DataFrame(
        [
            [9, 11, 11, 10],
            [2, 1, 2, 1],
            [10, 11, 11, 10],
            [3, 1, 2, 1],
        ],
        columns=markers,
    )
    fsom = fs.FlowSOM(data, cols_to_use=markers, n_clusters=2, replace=True, xdim=2, ydim=2, seed=42)
    assert fsom.get_cell_data().shape == (4, 4)
    assert fsom.get_cluster_data().shape == (4, 4)
    np.testing.assert_array_equal(
        fsom.get_cluster_data().X,
        np.array(
            [
                # 4 clusters, coinciding with the 4 cells
                [2.0, 1.0, 2.0, 1.0],
                [3.0, 1.0, 2.0, 1.0],
                [9.0, 11.0, 11.0, 10.0],
                [10.0, 11.0, 11.0, 10.0],
            ]
        ),
    )
    # the two metaclusters group cell 0 and 2, and cell 1 and 3
    pd.testing.assert_series_equal(
        fsom.get_cell_data().obs["metaclustering"],
        pd.Series([0, 1, 0, 1], name="metaclustering", index=pd.Index(range(4), dtype=str)),
    )
    # test MFI
    assert fsom.get_cluster_data().uns["metacluster_MFIs"].shape == (2, 4)
    assert fsom.get_cluster_data().uns["metacluster_MFIs"].columns.tolist() == markers
    # test values of MFI for clusters
    pd.testing.assert_frame_equal(
        fsom.get_cluster_data().uns["metacluster_MFIs"],
        pd.DataFrame(
            [
                # MFIs per cluster
                [9.5, 11.0, 11.0, 10.0],
                [2.5, 1.0, 2.0, 1.0],
            ],
            columns=markers,
            index=pd.Index([0, 1], name="metaclustering"),
        ),
    )
