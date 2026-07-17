import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Tuple

from skeleton_def import EDGES, JOINTS


# import matplotlib
# matplotlib.use("TkAgg")  # show plot in window back end
plt.switch_backend("agg")  # don't show plot


def get_palette(
    n: int, hue: float = 0.01, luminance: float = 0.6, saturation: float = 0.65
) -> np.array:
    hues = np.linspace(0, 1, n + 1)[:-1]
    hues += hue
    hues %= 1
    hues -= hues.astype(int)
    palette = [hls_to_rgb(float(hue), luminance, saturation) for hue in hues]
    palette = np.array(palette)
    return palette


def getDimBox(points):
    """
    points: ... N x DIM
    output:	[[min_d1, max_d1], ..., [min_dN, max_dN]]
    """
    if points.dtype == torch.float16:
        points = points.to(torch.float32)

    is_torch = torch.is_tensor(points[0])
    num_dim = points[0].shape[-1]
    if isinstance(points, list):
        assert is_torch or isinstance(points[0], np.ndarray)
        if is_torch:
            return np.array(
                [
                    [
                        np.median([pts[..., k].min().detach().cpu() for pts in points]),
                        np.median([pts[..., k].max().detach().cpu() for pts in points]),
                    ]
                    for k in range(num_dim)
                ]
            )
        else:
            return np.array(
                [
                    [
                        np.median([pts[..., k].min() for pts in points]),
                        np.median([pts[..., k].max() for pts in points]),
                    ]
                    for k in range(num_dim)
                ]
            )
    else:
        if is_torch:
            points = points.cpu().detach()
        if isinstance(points, np.ndarray):
            return np.array(
                [
                    [
                        np.median(points[..., k].min(-1)[0]),
                        np.median(points[..., k].max(-1)[0]),
                    ]
                    for k in range(num_dim)
                ]
            )
        else:
            return np.array(
                [
                    [
                        points[..., k].min(-1)[0].median(),
                        points[..., k].max(-1)[0].median(),
                    ]
                    for k in range(num_dim)
                ]
            )


def plot_pose(
    poses,
    connections=EDGES,
    azim=0,
    elev=0,
    save_fname="",
    legend=[],
    dim_box=None,
    title="",
    ax=None,
    is_blank=False,
    s=7,
    lw=3,
    dpi=300,
    show_axes=True,
    legend_fontsize=10,
    title_fontsize=30,
    is_title_below=False,
    black_edge=None,
    black_point=None,
    number_keypoints=False,
    colour="red",
):
    """
    poses : BATCH x NUM_PTS x 3
    connections: BATCH x EDGES or EDGES
    """

    # get colours
    colors_p = get_palette(n=poses.shape[-2])
    if isinstance(colors_p, np.ndarray) and (colors_p > 1).any():
        colors_p = [colors_p.copy() / 255]

    # get colours
    colors_l = get_palette(n=poses.shape[1])
    if isinstance(colors_l, np.ndarray) and (colors_l > 1).any():
        colors_l = [colors_l.copy() / 255]

    # create new plot
    if ax is None:
        plt.close("all")
        fig = plt.figure(figsize=(13.0, 20.0))
        # canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection="3d")
    else:
        if isinstance(ax, Axes3D):
            fig = ax.get_figure()
        else:
            fig = ax
            ax = fig.gca()

    # get bbox
    if dim_box is None:
        dim_box = getDimBox(poses)
    axes_line = [None for _ in range(len(poses))]

    # if not a batch add dim 0
    if len(poses.shape) == 2:
        poses = poses.unsqueeze(dim=0)

    for p, pose_ in enumerate(poses):
        s = [s for _ in range(len(pose_))]
        pose = pose_.cpu().detach().numpy() if torch.is_tensor(pose_) else pose_
        if pose.shape[0] == 176:
            ax.scatter(
                pose[:48, 0], pose[:48, 1], pose[:48, 2], color=colour, marker="o"
            )
            colour = "black"
            ax.scatter(
                pose[48:, 0], pose[48:, 1], pose[48:, 2], color=colour, marker="o", s=5
            )
        else:
            ax.scatter(
                pose[:, 0], pose[:, 1], pose[:, 2], color=colour, marker="o"
            )  # , s=s remove size from scatter
        # make one keypoint black
        if black_point is not None:
            ax.scatter(
                pose[black_point, 0],
                pose[black_point, 1],
                pose[black_point, 2],
                color="black",
                marker="x",
                s=50,
            )

        # plot connections if not None
        if connections is not None:
            connections_ = (
                connections[p] if isinstance(connections[0][0], tuple) else connections
            )
            for k, (x, y) in enumerate(connections_):
                color = colors_p[k]
                if k == black_edge:
                    color = np.array((1.0, 1.0, 1.0))
                axes_line[p] = ax.plot(
                    [pose[x, 0], pose[y, 0]],
                    [pose[x, 1], pose[y, 1]],
                    zs=[pose[x, 2], pose[y, 2]],
                    linewidth=lw,
                    color=color,
                )[0]
        if number_keypoints:
            for i in range(len(pose)):
                ax.text(pose[i, 0], pose[i, 1], pose[i, 2], str(i), fontsize=12)

    ax.set_xlim(dim_box[0])
    ax.set_ylim(dim_box[1])
    ax.set_zlim(dim_box[2])
    ax.set_box_aspect(
        (
            dim_box[0, 1] - dim_box[0, 0],
            dim_box[1, 1] - dim_box[1, 0],
            dim_box[2, 1] - dim_box[2, 0],
        )
    )
    if title is not None and title != "":
        if is_title_below:
            ax.set_title(title, fontsize=title_fontsize, y=-0.01)
        else:
            ax.set_title(title, fontsize=title_fontsize)
    ax.view_init(azim=azim, elev=elev)
    ax.legend(axes_line, legend, prop={"size": legend_fontsize})
    if is_blank:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        if show_axes:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.zaxis.set_ticks([])
        else:
            ax.axis("off")
        ax.grid(False)
    else:
        ax.set_xlabel("x", fontsize=25)
        ax.set_ylabel("y", fontsize=25)
        ax.set_zlabel("z", fontsize=25)
    if save_fname is not None and save_fname != "":
        plt.savefig(
            save_fname, bbox_inches="tight", pad_inches=0, dpi=dpi, transparent=True
        )
        print(f"Plot Saved: {save_fname}")
    # make plot tight
    plt.tight_layout()

    return fig


def make_pose_video(
    poses: list,
    names: list,
    video_name: str = "tri_pose_sequence",
    save_dir: Union[str, Path] = "./",
    fps: float = 50.0,
    slow: int = 1,
    main_title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Plot a sequence of 3d keypiont and plot them as a video.
    :param poses: 3 x F x J x 3
    :param names: 3 x name
    :param connections: skeleton connections
    :param video_name: name of video
    :param save_dir: dir to save
    :param fps: fps
    :param slow: slow down the video
    :param main_title: main title
    :param overwrite: overwite the video and frame
    :return: None
    """
    save_dir = Path(save_dir)
    if (save_dir / f"{video_name}.avi").exists() and not overwrite:
        print(f"Video already exists: {save_dir}")
        return

    poses, names = formate_poses(poses, names)

    # Create output video writer
    image_path = save_dir / video_name
    image_path.mkdir(parents=True, exist_ok=True)

    # print(f"Writing images to: {image_path}")
    n_plots = len(poses)

    # make constant bbox
    dim_boxes = [getDimBox(p) for p in poses]

    for frame_num, p in enumerate(
        tqdm(zip(*poses), desc="Plotting poses", total=len(poses[0]))
    ):
        # for frame_num, p in enumerate(zip(*poses)):
        if (image_path.absolute() / f"{frame_num}.jpeg").exists():
            continue
        fig, axs = plt.subplots(
            1, n_plots, figsize=(8 * n_plots, 8), subplot_kw=dict(projection="3d")
        )
        # plt.tight_layout()
        for i in range(n_plots):
            if isinstance(names[i], list):
                n = names[i][frame_num]
            else:
                n = names[i]

            n = n.replace("$", "")
            plot_pose(
                p[i].unsqueeze(0),
                connections=EDGES,
                title=n,
                ax=axs[i] if n_plots > 1 else axs,
                azim=-80,
                elev=-100,
                show_axes=False,
                is_blank=True,
                title_fontsize=15,
                dim_box=dim_boxes[i],
            )
        fig.suptitle(main_title.replace(".", "").replace("$", ""), fontsize=20)
        # reduce fig size
        # fig.set_size_inches(6.5, 3)
        fig.savefig(image_path.absolute() / f"{frame_num}.jpeg")
        plt.close(fig)

    imagedir2video(image_path, save_dir, slow=slow, name=video_name, fps=fps)

    remove_files(image_path)


def make_square_pose_video(
    poses: list,
    names: list,
    width: int,
    height: int,
    connections: list = None,
    video_name: str = "tri_pose_sequence",
    save_dir: Union[str, Path] = "./",
    fps: float = 50.0,
    slow: int = 1,
    main_title: str = "",
    overwrite: bool = False,
) -> None:
    """
    Plot a sequence of 3d keypiont and plot them as a video.
    for plotting the codebook make a W by H grid of poses
    :param poses: 3 x F x J x 3
    :param names: 3 x name
    :param connections: skeleton connections
    :param video_name: name of video
    :param save_dir: dir to save
    :param fps: fps
    :param slow: slow down the video
    :param main_title: main title
    :param overwrite: overwite the video and frame
    :return: None
    """
    save_dir = Path(save_dir)
    if (save_dir / f"{video_name}.avi").exists() and not overwrite:
        print(f"Video already exists: {save_dir}")
        return

    poses, names = formate_poses(poses, names)

    # Create output video writer
    image_path = save_dir / video_name
    image_path.mkdir(parents=True, exist_ok=True)

    # print(f"Writing images to: {image_path}")
    n_plots = len(poses)

    # make constant bbox
    dim_boxes = [getDimBox(p) for p in poses]

    for frame_num, p in enumerate(
        tqdm(zip(*poses), desc="Plotting poses", total=len(poses[0]))
    ):
        # for frame_num, p in enumerate(zip(*poses)):
        if (image_path.absolute() / f"{frame_num}.jpeg").exists():
            continue
        fig, axs = plt.subplots(
            width,
            height,
            figsize=(8 * width, 8 * height),
            subplot_kw=dict(projection="3d"),
        )
        i = 0
        for x in range(width):
            for y in range(height):
                i = (x * width) + y
                if isinstance(names[i], list):
                    n = names[i][frame_num]
                else:
                    n = names[i]

                n = n.replace("$", "")
                plot_pose(
                    p[i].unsqueeze(0),
                    connections=EDGES,
                    title=n,
                    ax=axs[x][y] if n_plots > 1 else axs,
                    azim=-80,
                    elev=-100,
                    show_axes=False,
                    is_blank=True,
                    title_fontsize=15,
                    dim_box=dim_boxes[i],
                )
        fig.suptitle(main_title.replace(".", "").replace("$", ""), fontsize=20)
        # change fig resolution in pixels
        fig.set_size_inches(30, 30)
        fig.savefig(image_path.absolute() / f"{frame_num}.jpeg")
        plt.close(fig)

    imagedir2video(image_path, save_dir, slow=slow, name=video_name, fps=fps)

    remove_files(image_path)


def formate_poses(poses: List[torch.Tensor], names: List[str]) -> Tuple:
    # reshape if needed
    _poses = []
    for p in poses:
        if p.shape[-1] != 3:
            p = p.reshape(p.shape[0], -1, 3)
        _poses.append(p)
    poses = _poses
    del _poses

    # extend the short
    max_len = max([p.shape[0] for p in poses])
    for i, p in enumerate(poses):
        if p.shape[0] < max_len:
            diff = max_len - p.shape[0]
            if diff > 0:
                p = torch.cat((p, p[-1].unsqueeze(0).repeat(diff, 1, 1)), dim=0)
                poses[i] = p
                if isinstance(names[i], list):
                    [names[i].append("<PAD>") for _ in range(diff)]
    return poses, names


def imagedir2video(
    pose_plot_path: Union[str, Path],
    save_dir: Union[str, Path],
    slow: int = 1,
    fps: int = 25,
    name: str = "",
):
    import cv2
    from PIL import Image

    """Takes a directory of images and creates a video from them."""
    pose_plot_path = Path(pose_plot_path)
    # read all images in the folder
    sort_key = lambda x: int(x.stem.split(".")[0])
    pose_images = [img for img in pose_plot_path.glob("*.jpeg") if img.is_file()]
    pose_images.sort(key=sort_key)

    # create video writer
    video_path = Path(save_dir) / f"{name}.avi"
    if video_path.is_file():
        print(f"Video already exists: {video_path} Overwritting")
        # return
    img = Image.open(pose_images[0])
    img = np.array(img)
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videoOut = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i, pose in enumerate(pose_images):
        try:
            pose = Image.open(pose)
            pose = pose.convert("RGB")
            pose = np.array(pose)
            frame = pose.astype(np.uint8)
            # convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(slow):
                videoOut.write(frame)
        except:
            videoOut.release()
        if i > 425:
            break

    videoOut.release()


def remove_files(image_path: Union[str, Path]) -> None:
    # remove image and folder
    [
        os.remove((image_path / file))
        for file in os.listdir(str(image_path))
        if file.endswith(".jpeg")
    ]
    os.rmdir(image_path)


def plot_codebook_pca(
    embeddings: torch.Tensor,
    codebook: torch.Tensor,
    epoch: int,
    save_dir: Union[str, Path],
):
    from sklearn.decomposition import PCA

    save_path = Path(save_dir) / "PCA"
    save_path.mkdir(parents=True, exist_ok=True)
    embeddings = torch.flatten(embeddings, -2, -1)
    embeddings = embeddings.detach().cpu().numpy()
    codebook = codebook.detach().cpu().numpy()

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embeddings)
    codebook_pca = pca.transform(codebook)

    fig = plt.figure(figsize=(10.0, 6.0))
    ax = fig.add_subplot(111)
    ax.scatter(
        principal_components[:, 0], principal_components[:, 1], label="Embeddings"
    )
    ax.scatter(codebook_pca[:, 0], codebook_pca[:, 1], label="Codebook")
    ax.legend()
    ax.set_title("PCA of embeddings and codebook")
    fig.savefig(save_path / f"PCA_{epoch}.jpeg")
    plt.close(fig)


def plot_codebook_usage(usage: torch.Tensor, epoch: int, save_dir: Union[str, Path]):
    """
    Plots the codebook usage
    :param usage: [N]
    :param epoch:  current epoch
    :param save_dir: Path to save the plot
    :return: None
    """
    save_dir = Path(save_dir)
    save_dir = save_dir / "Codebook_Usage"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = save_dir / f"{epoch}.jpeg"
    fig = plt.figure(figsize=(10.0, 6.0))
    ax = fig.add_subplot(111)
    ax.plot(usage.detach().cpu())

    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Usage")
    ax.set_title("Codebook usage")

    fig.savefig(save_dir)
    plt.close(fig)


def plot_confusion_matrix(
    cm,
    labels,
    title="Confusion Matrix",
    figsize=(8, 6),
    cmap="Blues",
    annot=False,
    fmt="d",
    save_path: str = None,
):
    """
    Plots a confusion matrix with labels and optional annotations.

    Args:
        cm: The confusion matrix, a 2D numpy array.
        labels: List of class labels (strings) corresponding to the matrix rows/columns.
        title: Title for the plot (default: 'Confusion Matrix').
        figsize: Figure size (width, height) in inches (default: (8, 6)).
        cmap: Colormap name for the heatmap (default: 'Blues').
        annot: Whether to display values in each cell (default: True).
        fmt: Format specifier for annotations (default: 'd' for integers).
    """

    # Create figure and axes
    plt.figure(figsize=figsize)
    ax = plt.gca()  # Get current axes

    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, ax=ax, yticklabels=labels)
    # xticklabels=[f'{i}' for i in range(cm.shape[1])]

    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Selected CB Index", fontsize=12)
    ax.set_ylabel("Gloss Labels", fontsize=12)

    # Improve readability (adjust as needed)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    if save_path is not None:
        plt.savefig(str(save_path))
