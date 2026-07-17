import torch
import numpy as np

from tqdm import tqdm
from scipy.signal import butter, filtfilt

from skeleton_def import JOINTS
from helpers import alternate_join_list


def stitch_poses(poses: torch.Tensor, stitch_config):
    """
    Stitch poses sequence together
    :param poses: list of [N x C x T]
    :param stitch_config: configuration for stitching
    """

    stitch_parts = []
    total_length = sum([d.shape[0] for d in poses])
    for x in range(len(poses) - 1):
        sign_a = poses[x]
        sign_b = poses[x + 1]

        connection = stitch_ab(sign_a, sign_b, stitch_config)
        stitch_parts.append(connection)
    stitched_seq = alternate_join_list(poses, stitch_parts)
    stitched_seq = torch.cat(stitched_seq, dim=0)
    if stitch_config.get("resample_seq", False):
        # resample the sequence
        total_length = int(
            total_length * float(stitch_config.get("seq_resample_factor", 1))
        )
        stitched_seq = interpolate_pose(stitched_seq, total_length)
    if stitch_config.get("lp_cutoff", 0) > 0:
        # apply low pass filter
        stitched_seq = apply_low_pass_filter(
            stitched_seq, cutoff_freq=stitch_config["lp_cutoff"]
        )
        # stitched_sequences.append(stitched_seq)
    return stitched_seq


def stitch_ab(sign_a: torch.Tensor, sign_b: torch.Tensor, model_config):
    stitch_length = determine_stitch_length(sign_a, sign_b, "pose")
    stitch = interpolate_between_poses(sign_a, sign_b, stitch_length)
    return stitch


def interpolate_between_poses(
    sign_a: torch.Tensor, sign_b: torch.Tensor, n_frames: int
):
    """
    interpolate between two poses, start and end pose
    :param sign_a: start pose sign
    :param sign_b: end pose sign
    :param n_frames: the number of frame to stitch together with
    :return:
    """
    interp_size = calculate_interpolate_size(n_frames)
    stitch = torch.nn.functional.interpolate(
        torch.stack([sign_a[-1], sign_b[0]]).permute(2, 1, 0),
        size=interp_size,
        mode="linear",
    ).permute(2, 1, 0)

    for s in range(stitch.shape[0]):
        if not torch.equal(sign_a[-1], stitch[s]):
            break
    for e in reversed(range(stitch.shape[0])):
        if not torch.equal(sign_b[0], stitch[e]):
            break
    # Stop index error
    e += 1 if e + 1 < stitch.shape[0] else e

    stitch = stitch[s:e]
    assert stitch.shape[0] == n_frames, "Stitch is not the correct length"
    return stitch


def determine_stitch_length(sign_a: torch.Tensor, sign_b: torch.Tensor, data_type: str):
    # select right hand to track
    if data_type == "pose":
        track_point_a = sign_a[..., JOINTS["RWrist"], :]
        track_point_b = sign_b[..., JOINTS["RWrist"], :]
    else:
        raise ValueError(f"Unknown data type {data_type}")

    cadidates = []
    for i in range(1, 15):
        c = interpolate_between_poses(sign_a, sign_b, i)
        cadidates.append(c)

    track_point_a = track_point_a[-3:]
    track_point_b = track_point_b[:3]

    if data_type == "pose":
        track_point_c = [c[..., JOINTS["RWrist"], :] for c in cadidates]
    else:
        raise ValueError(f"Unknown data type {data_type}")

    vel_a = traj_to_acceleration(traj=track_point_a[-2:].squeeze(-1))[1]
    vel_b = traj_to_acceleration(traj=track_point_b[:2].squeeze(-1))[1]
    c_mean = [
        torch.mean(torch.Tensor(traj_to_acceleration(traj=c.squeeze(-1))[1]))
        for c in track_point_c
    ]
    try:
        if len(vel_a) == 0 and len(vel_b) == 0:
            return 3
        elif len(vel_a) == 0:
            threshold = max(vel_b).item()
        elif len(vel_b) == 0:
            threshold = max(vel_a).item()
        else:
            threshold = max(vel_a, vel_b).item()
    except:
        print("Error calculating velocity")
    # find the first index where the velocity is greater than the threshold
    for use_n_frame, c in enumerate(c_mean):
        if c < threshold:
            break
    use_n_frame += 1

    return use_n_frame


def traj_to_acceleration(traj: np.array = None):
    """
    Convert trajectory to acceleration
    :param traj: [N x 3]
    :return:
    """
    # get acceleration
    traj = np.abs(traj)

    velocity = np.diff(traj, axis=0)
    velocity = np.abs(velocity)
    velocity = np.sum(velocity, axis=-1)

    acceleration = np.diff(velocity, axis=0)
    acceleration = np.abs(acceleration)

    return acceleration, velocity


def interpolate_pose(poses: torch.Tensor = None, num_sample_pts: int = 40):
    """
    Interpolates a pose sequence using linear interpolation
    :param poses: N x K x 3
    :param num_sample_pts: number of points to sample
    :return: interpolated pose
    """
    # print('Interpolating Pose')
    squeeze = False
    if len(poses.shape) == 2:
        poses = poses.unsqueeze(dim=-1)
        squeeze = True

    new_pose = torch.nn.functional.interpolate(
        poses.permute(2, 1, 0), size=num_sample_pts, mode="linear"
    ).permute(2, 1, 0)
    if squeeze:
        new_pose = new_pose.squeeze(dim=-1)

    return new_pose


def calculate_interpolate_size(stitch_size):
    """
    Calculates the corresponding interpolate size based on the stitch size

    Args:
        stitch_size (int): The input stitch size.

    Returns:
        int: The calculated interpolate size.
    """
    # Check if the stitch size is an odd number.
    if stitch_size % 2 != 0:
        # If odd, the relationship is y = 2x + 1.
        return (2 * stitch_size) + 1
    # Check if the stitch size is one of the special even cases.
    elif stitch_size in [0, 28, 42]:
        # For these specific even values, the relationship is y = 2x + 2.
        return (2 * stitch_size) + 2
    # This is the default case for all other even numbers.
    else:
        # For other even values, the relationship is y = 2x.
        return 2 * stitch_size


def apply_low_pass_filter(pose_sequence, cutoff_freq=15, fs=60):
    """
    Applies a low-pass filter to smooth the movement in a MediaPipe pose sequence.

    Args:
        pose_sequence (ndarray): MediaPipe pose sequence of shape (N, 61, 3).
        cutoff_freq (float): Cutoff frequency for the low-pass filter (in Hz). Default is 5.
        fs (float): Sampling frequency of the pose sequence (in Hz). Default is 30.

    Returns:
        ndarray: Smoothed pose sequence of the same shape as the input.

    """
    # Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / (0.5 * fs)

    # Create low-pass Butterworth filter coefficients
    b, a = butter(4, normalized_cutoff, btype="low", analog=False)

    # Apply the filter along each dimension
    smoothed_sequence = np.apply_along_axis(
        lambda x: filtfilt(b, a, x, padtype=None), axis=0, arr=pose_sequence
    )
    return torch.Tensor(smoothed_sequence).clone().detach()
