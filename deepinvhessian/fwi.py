import torch
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from scipy.ndimage import gaussian_filter
from scipy import signal
from tqdm.notebook import tqdm
from typing import Callable, Optional
from deepinvhessian.utilities import *
from deepinvhessian.filters import lowpass_filter


class FWIParams:
    """
    A class to hold parameters and functions for Full Waveform Inversion (FWI).

    This class initializes and stores the parameters required for FWI and provides
    methods to calculate source and receiver coordinates, as well as to create wavelets.

    Attributes
    ----------
    nx : int
        Number of grid points in the x-direction.
    nz : int
        Number of grid points in the z-direction.
    dx : float
        Grid spacing in the x-direction.
    nt : int
        Number of time samples.
    dt : float
        Time sampling interval.
    num_dims : int
        Number of dimensions.
    num_shots : int
        Number of shots.
    num_batches : int
        Number of batches.
    num_sources_per_shot : int
        Number of sources per shot.
    num_receivers_per_shot : int
        Number of receivers per shot.
    ds : int or torch.Tensor
        Spacing between sources.
    dr : int or torch.Tensor
        Spacing between receivers.
    sz : torch.Tensor
        Depth of sources.
    rz : torch.Tensor
        Depth of receivers.
    os : float
        Offset of sources.
    orec : float
        Offset of receivers.
    ox : float
        Offset in the x-direction.
    freq : float
        Dominant frequency of the wavelet.
    wavelet : torch.Tensor
        Source wavelet tensor.
    s_cor : torch.Tensor
        Source locations [num_shots, num_sources_per_shot, num_dimensions].
    r_cor : torch.Tensor
        Receiver locations [num_shots, num_receivers_per_shot, num_dimensions].

    Methods
    -------
    get_coordinate(acquisition: int | str) -> tuple[torch.Tensor, torch.Tensor]
        Calculate source and receiver coordinates based on the acquisition mode.
    create_wavelet(wavelet: torch.Tensor, scale: float = 1.0) -> torch.Tensor
        Create a tensor of source amplitudes from the wavelet values with an optional scale.
    """

    def __init__(self, par: dict, wavelet: torch.Tensor, acquisition: int | str):
        """
        Initialize an instance to apply FWI.

        Parameters
        ----------
        par : dict
            Dictionary containing all the parameters of the models used to apply the inversion.
        wavelet : torch.Tensor
            Source wavelet tensor.
        acquisition : int | str
            Type of acquisition. Options are:
            1: Receivers are spread over the whole surface.
            2: Specific offset for receivers.
            'volve': Custom mode for handling Volve dataset.
        """
        # Unpacking and storing parameters
        self.nx = par['nx']
        self.nz = par['nz']
        self.dx = par['dx']
        self.nt = par['nt']
        self.dt = par['dt']
        self.num_dims = par['num_dims']
        self.num_shots = par['num_shots']
        self.num_batches = par['num_batches']
        self.num_sources_per_shot = par['num_sources_per_shot']
        self.num_receivers_per_shot = par['num_receivers_per_shot']
        self.ds = par['ds']
        self.dr = par['dr']
        self.sz = par['sz']
        self.rz = par['rz']
        self.os = par['os']
        self.orec = par['orec']
        self.ot = par['ot']
        self.ox = par['ox']
        self.freq = par['freq']

        self.s_cor, self.r_cor = self.get_coordinate(acquisition)
        self.source_amplitudes = self.create_wavelet(wavelet)



    def get_coordinate(self, mode: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Create arrays containing the source and receiver locations based on the mode.

        Parameters:
        ----------
        mode : int | str
            - 1: Receivers are spread over the whole surface.
            - 2: Specific offset for receivers.
            - 'volve': Custom mode for handling Volve dataset.

        Returns:
        -------
        tuple[torch.Tensor, torch.Tensor]
            Source and Receiver locations:
            - Source locations [num_shots, num_sources_per_shot, num_dimensions].
            - Receiver locations [num_shots, num_receivers_per_shot, num_dimensions].

        """
        x_s = torch.zeros(self.num_shots, self.num_sources_per_shot, self.num_dims)
        x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
        # x direction
        x_s[:, 0, 1] = torch.arange(0, self.num_shots).float() * self.ds + self.os - self.ox
        # z direction
        x_s[:, 0, 0] = self.sz

        if mode == 1:
            # x direction
            x_r[0, :, 1] = torch.arange(0, self.num_receivers_per_shot).float() * self.dr + self.orec
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
            # z direction
            x_r[:, :, 0] = self.rz
            

        elif mode == 2:  # fixed spread !!
            # x direction
            x_r[0, :, 1] = torch.arange(self.num_receivers_per_shot).float() * self.dr + self.orec
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1) + \
                           torch.arange(0, self.num_shots).repeat(self.num_receivers_per_shot, 1).T * self.ds - self.ox
            # z direction
            x_r[:, :, 0] = self.rz
        
        elif mode == 'volve_synthetic':
            x_s = torch.cat([self.sz.reshape(-1, 1), self.ds.reshape(-1, 1)], dim=1).reshape(-1, 1, 2)

            x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
            x_r[0, :, 1] = self.dr
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
            x_r[0, :, 0] = self.rz
            for idx in range(self.num_shots):
                x_r[idx, :, 0] = self.rz[idx]
            x_s[:,:,1] = x_s[:,:,1]
            x_r[:,:,1] = x_r[:,:,1]
        
        elif mode == 'volve':
            x_s = torch.cat([self.sz.reshape(-1, 1), self.ds.reshape(-1, 1)], dim=1).reshape(-1, 1, 2)

            x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
            x_r[0, :, 1] = self.dr
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
            x_r[0, :, 0] = self.rz
            for idx in range(self.num_shots):
                x_r[idx, :, 0] = self.rz[idx]
            x_s[:,:,1] = x_s[:,:,1] - 2800
            x_r[:,:,1] = x_r[:,:,1] - 2800
        
        x_s /= self.dx
        x_r /= self.dx

        return x_s, x_r
    
    def create_wavelet(self, wavelet: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Creates a scaled wavelet tensor for source amplitudes from the given wavelet values.

        Parameters
        ----------
        wavelet : torch.Tensor
            A tensor of wavelet values that represent the seismic signal to be emitted by the sources.
        scale : float, optional
            A scaling factor applied to the wavelet to adjust its amplitude. Default is 1.0.

        Returns
        -------
        torch.Tensor
            A tensor shaped [num_shots, 1, num_sources_per_shot], representing the scaled source amplitudes.
        """
        source_amplitudes = scale * wavelet.reshape(1, 1, -1).repeat(self.num_shots, 1, self.num_sources_per_shot).float()
        return source_amplitudes


    def create_masks(self, window_size: int = 600, v_direct: float = 1800, more_near_offset_mute: int = None, **kwargs) -> np.ndarray:
        """
        Create masks for seismic data processing with optional near offset muting.

        :param window_size: int, optional
            The size of the window for muting, default is 600 samples.
        :param v_direct: float, optional
            Velocity of the direct wave, default is 1800 m/s.
        :param more_near_offset_mute: int, optional
            Additional muting near the source-receiver offset, specified as the width of the taper to apply.
        :return: np.ndarray
            A 3D array of masks for each shot and receiver with applied muting and tapering.
        """
        masks = np.ones((self.num_shots, self.num_receivers_per_shot, self.nt))
        if torch.is_tensor(self.ds):
            ds = np.round(self.ds[1] - self.ds[0]).item()  # Assuming ds is a torch.Tensor, convert to Python float
            dr = np.round(self.dr[1] - self.dr[0]).item()  # Assuming dr is a torch.Tensor, convert to Python float
        else:
            ds, dr = self.ds, self.dr  # Use directly if not tensors

        ot = kwargs.get('ot', self.ot)  # Get 'ot' from kwargs or use default

        for shot_idx in range(self.num_shots):
            sx = (shot_idx * ds + self.os)
            for receiver_idx in range(self.num_receivers_per_shot):
                rx = (receiver_idx * dr + self.orec)
                dist = abs(sx - rx)
                arrival_time = dist / v_direct / self.dt + ot
                window_start = int(arrival_time) - window_size // 2
                window_end = window_start + window_size

                actual_window_start = max(window_start, 0)
                actual_window_end = min(window_end, self.nt)

                masks[shot_idx, receiver_idx, :actual_window_start] = 0  # Mute before the window

                taper_length = actual_window_end - actual_window_start
                if taper_length > 0:
                    taper = (1 - np.cos(np.linspace(0, np.pi, taper_length))) / 2
                    masks[shot_idx, receiver_idx, actual_window_start:actual_window_end] = taper

                if more_near_offset_mute is not None and abs(shot_idx - receiver_idx) <= 10:
                    taper_t = np.ones(self.nt)
                    width = more_near_offset_mute
                    taper_t[:taper_length+width] = (1 - np.cos(np.linspace(0, np.pi, taper_length+width))) / 2
                    masks[shot_idx, receiver_idx, :] *= taper_t

        return masks

    
def forward_modelling(params: FWIParams, model: torch.Tensor,  device: str):
    """2D acoustic wave equation forward modeling.

    Parameters:
    - model (torch.Tensor): Model tensor.
    - wavelet (torch.Tensor): Wavelet tensor.
    - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

    Returns:
    - data (torch.Tensor): Seismic data.

    """
    # pml_width parameter control the boundary, for free surface first argument should be 0
    data = deepwave.scalar(model.to(device), params.dx, params.dt,
        source_amplitudes= params.source_amplitudes.to(device),
        source_locations=params.s_cor.to(device),
        receiver_locations=params.r_cor.to(device),
        pml_width=[20, 20, 20, 20],
        accuracy=8,
        # pml_freq=params.freq,
    )[-1]
    return data


def forward_modelling_born(params: FWIParams, model: torch.Tensor, scalar: torch.Tensor,  device: str):
    """2D acoustic wave equation forward modeling.

    Parameters:
    - model (torch.Tensor): Model tensor.
    - wavelet (torch.Tensor): Wavelet tensor.
    - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

    Returns:
    - data (torch.Tensor): Seismic data.

    """
    # pml_width parameter control the boundary, for free surface first argument should be 0
    data = deepwave.scalar_born(model.to(device), scalar.to(device), params.dx, params.dt,
        source_amplitudes= params.source_amplitudes.to(device),
        source_locations=params.s_cor.to(device),
        receiver_locations=params.r_cor.to(device),
        pml_width=[20, 20, 20, 20],
        accuracy=8,
        # pml_freq=params.freq,
    )[-1]
    return data

def compute_gradient(
    params: FWIParams, 
    model: torch.Tensor, 
    observed_data: torch.Tensor, 
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ignore_n_samples: int, 
    device: str,
    mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, float]:
    """
    通过反向传播计算模型参数的梯度，可选地对数据应用掩膜。

    参数说明：
    params : FWIParams
        包含反演相关参数（如dx、dt、源/检波器位置等）的对象。
    model : torch.Tensor
        当前速度模型张量，需要计算梯度。
    observed_data : torch.Tensor
        实际观测到的地震数据。
    loss_function : Callable
        用于计算预测数据与观测数据之间误差的损失函数。
    ignore_n_samples : int
        忽略前若干个时间采样点（如直达波）。
    device : str
        指定运行设备（'cuda'或'cpu'）。
    mask : Optional[torch.Tensor]
        可选的掩膜张量，用于数据加权或屏蔽（shape需与数据一致）。

    返回值：
    gradient : torch.Tensor
        当前模型对应的梯度张量（克隆）。
    running_loss : float
        所有batch的平均loss。
    """

    # 将模型和掩膜（如果有）转移至目标计算设备
    model = model.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # 初始化累计loss
    running_loss = 0.0
    num_batches = params.num_batches
    
    # 清零已有梯度，避免累加
    if model.grad is not None:
        model.grad.zero_()
    
    # 遍历所有batch（或shot）
    for it in range(num_batches):
        # Simulate the wave propagation using the current model and batch of sources and receivers   # 使用当前模型和源/接收器信息进行波动方程正演模拟
        batch_data_pred = deepwave.scalar(
            model,
            params.dx,
            params.dt,
            source_amplitudes=params.source_amplitudes[it::num_batches].to(device),
            source_locations=params.s_cor[it::num_batches].to(device),
            receiver_locations=params.r_cor[it::num_batches].to(device),
            pml_width=[20, 20, 20, 20],
            accuracy=8,
            pml_freq=params.freq
        )[-1]
        
        # 如果使用掩膜，按batch维度加权后再对早期时间窗裁剪
        if mask is not None:
            batch_data_pred *= mask[it::num_batches]
            batch_data_pred = torch.nn.functional.pad(batch_data_pred[:, :, ignore_n_samples:], (0, ignore_n_samples, 0, 0, 0, 0), 'constant', 0)

        # 获取对应的观测数据batch
        batch_observed_data = observed_data[it::num_batches].to(device)

        # 计算预测数据与观测数据之间的loss
        batch_loss = loss_function(
            batch_data_pred.squeeze(), 
            batch_observed_data.squeeze()
        )
        
        # 反向传播，计算当前模型的梯度
        batch_loss.backward()
        
        # 累加当前batch的loss
        running_loss += batch_loss.item()
        
     # 返回梯度副本（避免被后续操作修改）和平均loss
    return model.grad.clone(), running_loss / num_batches


def compute_gradient_born(
    params: FWIParams, 
    model_true: torch.Tensor, 
    model: torch.Tensor, 
    observed_data: torch.Tensor, 
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ignore_n_samples: int, 
    device: str,
    mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, float]:
    """
    通过反向传播计算模型参数的梯度，可选地对数据应用掩膜。

    参数说明：
    params : FWIParams
        包含反演相关参数（如dx、dt、源/检波器位置等）的对象。
    model : torch.Tensor
        当前速度模型张量，需要计算梯度。
    observed_data : torch.Tensor
        实际观测到的地震数据。
    loss_function : Callable
        用于计算预测数据与观测数据之间误差的损失函数。
    ignore_n_samples : int
        忽略前若干个时间采样点（如直达波）。
    device : str
        指定运行设备（'cuda'或'cpu'）。
    mask : Optional[torch.Tensor]
        可选的掩膜张量，用于数据加权或屏蔽（shape需与数据一致）。

    返回值：
    gradient : torch.Tensor
        当前模型对应的梯度张量（克隆）。
    running_loss : float
        所有batch的平均loss。
    """

    # 将模型和掩膜（如果有）转移至目标计算设备
    model = model.to(device)
    model_true = model_true.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # 初始化累计loss
    running_loss = 0.0
    num_batches = params.num_batches
    
    # 清零已有梯度，避免累加
    if model.grad is not None:
        model.grad.zero_()
    
    # 遍历所有batch（或shot）
    for it in range(num_batches):
        # Simulate the wave propagation using the current model and batch of sources and receivers   # 使用当前模型和源/接收器信息进行波动方程正演模拟
        batch_data_pred = deepwave.scalar_born(
            model_true,
            model,
            params.dx,
            params.dt,
            source_amplitudes=params.source_amplitudes[it::num_batches].to(device),
            source_locations=params.s_cor[it::num_batches].to(device),
            receiver_locations=params.r_cor[it::num_batches].to(device),
            pml_width=[20, 20, 20, 20],
            accuracy=8,
            pml_freq=params.freq
        )[-1]
        
        # 如果使用掩膜，按batch维度加权后再对早期时间窗裁剪
        if mask is not None:
            batch_data_pred *= mask[it::num_batches]
            batch_data_pred = torch.nn.functional.pad(batch_data_pred[:, :, ignore_n_samples:], (0, ignore_n_samples, 0, 0, 0, 0), 'constant', 0)

        # 获取对应的观测数据batch
        batch_observed_data = observed_data[it::num_batches].to(device)

        # 计算预测数据与观测数据之间的loss
        batch_loss = loss_function(
            batch_data_pred.squeeze(), 
            batch_observed_data.squeeze()
        )
        
        # 反向传播，计算当前模型的梯度
        batch_loss.backward()
        
        # 累加当前batch的loss
        running_loss += batch_loss.item()
        
     # 返回梯度副本（避免被后续操作修改）和平均loss
    return model.grad.clone(), running_loss / num_batches


def compute_dm1(
    params: FWIParams, 
    model: torch.Tensor, 
    dm1: torch.Tensor, 
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ignore_n_samples: int, 
    device: str,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
        使用 Born 建模和反向传播计算散射扰动项 dm1 的梯度，并可选择应用数据掩膜。

        参数说明：
        ----------
        params : FWIParams
            存储波动方程建模所需的参数对象（如源、接收器、时间步长等）。
        model : torch.Tensor
            背景速度模型（背景场）。
        dm1 : torch.Tensor
            一阶散射扰动项，表示相对于背景模型的微小扰动。
        loss_function : Callable
            损失函数，通常计算预测散射波与“零数据”之间的误差。
        ignore_n_samples : int
            忽略前若干时间采样点（常用于去除直达波影响）。
        device : str
            使用的设备，如 'cuda' 或 'cpu'。
        mask : Optional[torch.Tensor]
            可选掩膜，用于加权或屏蔽预测数据中的部分区域。

        返回：
        -------
        dm1_grad : torch.Tensor
            与散射扰动项 dm1 相对应的梯度张量。
    """


    # 将模型、扰动项以及掩膜（如果有）转移到目标计算设备上
    model = model.to(device)
    dm1 = dm1.to(device)
    if mask is not None:
        mask = mask.to(device)

    # 若 dm1 尚未关联梯度张量，则初始化一个全零张量
    if dm1.grad is None:
        dm1.grad = torch.zeros_like(dm1)
    # 获取批次数（即每次反演中的shot数）
    num_batches = params.num_batches

    # Zero gradients before the loop
    # dm1.grad.zero_()

    for it in range(num_batches):   # 遍历每个 shot（源点）
        # 使用 Born 建模方法，模拟当前扰动下的散射波场
        batch_data_pred = deepwave.scalar_born(
            model,
            dm1,
            params.dx,
            params.dt,
            source_amplitudes=params.source_amplitudes[it::num_batches].to(device),
            source_locations=params.s_cor[it::num_batches].to(device),
            receiver_locations=params.r_cor[it::num_batches].to(device),
            pml_width=[20, 20, 20, 20],
            accuracy=8,
            pml_freq=params.freq,
        )[-1].squeeze()
        
        if mask is not None:
            batch_data_pred *= mask[it::num_batches].squeeze()

         # 如果提供掩膜，则对模拟结果加权（如屏蔽部分接收器或区域）
        batch_loss = loss_function(batch_data_pred[:, ignore_n_samples:], torch.zeros_like(batch_data_pred[:, ignore_n_samples:]))
        # batch_loss = loss_function(batch_data_pred, torch.zeros_like(batch_data_pred))
        
        # 通过反向传播计算该损失对 dm1 的梯度
        batch_loss.backward()

     # 返回 dm1 的梯度副本（clone 以避免原始梯度被外部修改）
    return dm1.grad.clone()


def source_illumination(model: torch.Tensor, source: torch.Tensor, dx: float, dt: float, 
                        x_s: torch.Tensor, device: str) -> torch.Tensor:
    """
    Calculate the source illumination by simulating the wavefield for each shot and summing the energy.

    Parameters:
    - model (torch.Tensor): The velocity model.
    - source (torch.Tensor): Source wavelet.
    - dx (float): Spatial discretization.
    - dt (float): Time sampling interval.
    - x_s (torch.Tensor): Source locations.
    - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

    Returns:
    - src_illum (torch.Tensor): The source illumination pattern.
    """
    nz, nx = model.shape
    num_shots, _, nt = source.shape
    num_batches = num_shots

    x = torch.arange(nx, dtype=torch.float32) 
    z = torch.arange(nz, dtype=torch.float32) 
    x, z = torch.meshgrid(x, z, indexing='ij')
    x, z = x.flatten(), z.flatten()
    # 'Receivers' at every point in the model
    x_snap = torch.stack([z, x], dim=-1)
    x_snap = x_snap.unsqueeze(0).expand(num_shots, -1, -1)

    # Move the snapshot grid to the device
    x_snap = x_snap.to(device)

    result = torch.zeros((1, nz*nx, nt), device=device)

    # Simulate the wavefield for each batch and sum up the energy
    for it in tqdm(range(num_shots)):
        source_wavefield = deepwave.scalar(
            model.to(device), 
            dx, 
            dt,
            source_amplitudes=source[it::num_batches].to(device),
            source_locations=x_s[it::num_batches].to(device),
            receiver_locations=x_snap[it::num_batches],
            pml_width=[20, 20, 20, 20],
            accuracy=8,
        )[-1].squeeze()

        # Sum the squared wavefield to result
        result += source_wavefield ** 2
    
    # Sum over time to get the final source illumination
    src_illum = result.sum(dim=-1)
    src_illum = src_illum.squeeze().reshape(nx, nz).T

    return src_illum.to(device)

def process_data(
    params: FWIParams,
    data: np.ndarray, 
    pd: int, 
    fn: float, 
    time_shift: int, 
    window_size: int = 600, 
    v_direct: float = 1800, 
    more_near_offset_mute: Optional[int] = None,
) -> torch.Tensor:
    """
    Process seismic data with padding, tapering, filtering, and masking for Volve data.

    Parameters
    ----------
    params : FWIParams
        Object containing the parameters for computation.
    data: Input seismic data as a NumPy array.
    pd: Padding size.
    fn: Cutoff frequency for the lowpass filter.
    time_shift: Time shift applied to the data.
    window_size: Size of the window for masking, default is 600.
    v_direct: Direct wave velocity for masking, default is 1800 m/s.
    more_near_offset_mute: Additional muting near the source-receiver offset.
    
    :return: Processed data as a torch.Tensor.
    """
    
    # Ensure the time axis is the last one
    if data.shape[0] > data.shape[1]:
        data = np.transpose(data, (1, 2, 0))
    nt = data.shape[2]
    
    # Apply padding and tapering
    data_padded = np.pad(data, ((0, 0), (0, 0), (pd, pd)), mode='edge')
    time_taper = cosine_taper1d(nt + 2 * pd, top_width=pd, bottom_width=pd)
    data_padded *= time_taper
    
    # Apply lowpass filtering
    filtered_data = lowpass_filter(6, fn, data_padded, params.dt, filteringN=2)[..., pd:-pd]
    
    # Generate and apply masks to the direct arrivals
    masks = params.create_masks(window_size=window_size, v_direct=v_direct, more_near_offset_mute=more_near_offset_mute)
    observed_data = filtered_data * masks
    
    # Convert to torch.Tensor and return
    observed_data_tensor = torch.tensor(observed_data).float()
    return observed_data_tensor


def bb_step(deltaX: torch.Tensor, deltaG: torch.Tensor, step_type: str = 'short', epsilon: float = 1e-8) -> float:
    """
    Calculate the step size using the Barzilai-Borwein method with an epsilon for numerical stability.

    The Barzilai-Borwein method provides an estimation for the step size in gradient-based
    optimization algorithms. It attempts to approximate the inverse of the Hessian matrix
    using information from the previous step.

    Parameters:
    - deltaX (torch.Tensor): The difference between the current and previous solutions (x_k - x_{k-1}).
    - deltaG (torch.Tensor): The difference between the current and previous gradients (g_k - g_{k-1}).
    - step_type (str): Determines the type of Barzilai-Borwein step size to compute. It can be either
      'short' for the short-step or 'long' for the long-step. Default is 'short'.
    - epsilon (float): A small value added to the denominator for numerical stability. Default is 1e-8.

    Returns:
    - float: The computed step size as a scalar value.

    Raises:
    - ValueError: If `step_type` is not 'short' or 'long'.

    Example:
    >>> x_k = torch.tensor([1.0, 2.0])
    >>> x_km1 = torch.tensor([0.5, 1.5])
    >>> g_k = torch.tensor([0.1, 0.2])
    >>> g_km1 = torch.tensor([0.1, 0.1])
    >>> bb_step(x_k - x_km1, g_k - g_km1)
    10.0
    """
    if step_type not in ['short', 'long']:
        raise ValueError("step_type must be 'short' or 'long'")

    deltaX, deltaG = deltaX.flatten().float(), deltaG.flatten().float()
    if step_type == 'short':
        numerator = torch.dot(deltaX, deltaG)
        denominator = torch.dot(deltaG, deltaG)
    elif step_type == 'long':
        numerator = torch.dot(deltaX, deltaX)
        denominator = torch.dot(deltaX, deltaG)

    denominator = max(denominator, epsilon)

    alpha = numerator / denominator
    return alpha.item()

def run_fwi(params, model, data, optimizer, loss_fn, freq, FWI_iter, device, *, 
            clip_gradient=None, source_illumination=None, mask=None, taper=None, tsamples=0, bb_step_length=None,
            show_data=False, save_results, exp_name='FWI_exp'):
    model = model.to(device)
    model.requires_grad = True
    gradients, updates, fwi_loss, alphas = [], [], []
    for iteration in tqdm(range(FWI_iter)):
        # Save simulated data
        if iteration == 0 or iteration == FWI_iter - 1:
            data = forward_modelling(params, model.detach().clone(), device).cpu().numpy()
            np.savez(f'{exp_name}/simulated_data_iter_{iteration}_freq_{freq}_grad', data=data)
            if show_data:
                show_3_shots(data, [10, 90, 170], clip=0.02, extent=(params['dr'][0],params['dr'][-1], params['nt']*params['dt'], 0), 
                ylim=(params['nt']*params['dt'], 0), save_path=f'{exp_name}/simulated_data_iter_{iteration}_freq_{freq}.png')
            
        # Compute FWI gradient
        optimizer.zero_grad()
        grad, iter_loss = compute_gradient(params, model, data, loss_fn, tsamples, device)
        fwi_loss.append(iter_loss)
        print(f'FWI iteration: {iteration} loss = {iter_loss}')
        # Clip the gradient values
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_value_(model, torch.quantile(grad.detach().abs(), clip_gradient))
        # Apply source illumination to the gradient
        if source_illumination is not None:
            grad = (grad * model.detach().clone()**3 ) / source_illumination
        if iteration == 0: gmax0 =  torch.abs(grad.detach()).max()
        # Normalize the gradient, mask it around the sources and apply taperinh to the shallower and deeper parts
        grad = (grad /gmax0) * mask * taper
        if mask is not None:
            grad *= mask
        if taper is not None:
            grad *= taper
        gradients.append(grad.cpu().detach().numpy())
        if bb_step_length is not None:
            if iteration > 0:
                delta_model = model.detach().clone() - previous_model
                delta_grad = grad.detach().clone() - previous_grad
                alpha = bb_step(delta_model, delta_grad, 'short')
                optimizer.param_groups[-1]['lr'] = alpha
                alphas.append(alpha)
            # Save the current solution and gradient for calculating the step size in the next iteration
            previous_model = model.detach().clone()
            previous_grad = grad.detach().clone()
        # Update the model
        model.grad.data[:] = grad
        optimizer.step()
        updates.append(model.detach().clone().cpu().numpy())
    # Save the results
    if save_results is not None:
        np.savez(f'{exp_name}/losses_grad', fwi_loss=np.array(fwi_loss),)
        np.savez(f'{exp_name}/gradient', updates=np.array(updates), gradients=np.array(gradients), )
