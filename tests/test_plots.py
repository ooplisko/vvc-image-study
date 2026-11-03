from vvcstudy import plots


def test_partition_plot_uses_density_for_different_size_div2k_images(tmp_path, monkeypatch):
    captured = []
    real_close = plots.plt.close

    def capture(figure):
        if hasattr(figure, "axes"):
            axis = figure.axes[0]
            captured.append((axis.get_ylabel(), axis.collections[0].get_offsets()[:, 1].tolist()))
        real_close(figure)

    monkeypatch.setattr(plots.plt, "close", capture)
    features = ("sobel_si", "luma_sd", "edge_fraction", "glcm_contrast", "glcm_entropy", "glcm_homogeneity")
    rows = [{"dataset": dataset, "distortion": "clean", "qp": 32,
             "cu_count": count, "cu_density_per_mpixel": 50,
             **{name: i + 1 for name in features}}
            for dataset in ("div2k", "kodak") for i, count in enumerate((100, 200))]
    plots.partition_plots(rows, tmp_path)
    assert captured == [("Final luma CU per million coded pixels", [50, 50]),
                        ("Final luma CU count", [100, 200])]
