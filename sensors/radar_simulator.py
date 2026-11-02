from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sionna.rt import PathSolver


@dataclass
class RadarConfig:
    sensor_id: str = "radar_0"
    carrier_frequency: float = 77e9
    bandwidth: float = 200e6
    num_samples: int = 256
    num_chirps: int = 8
    t_pri: float = 40e-6
    interval_s: float = 0.1
    max_depth: int = 3
    samples_per_src: int = 100000
    max_num_paths_per_src: int = 10000
    seed: int = 41
    noise_snr_db: float | None = None
    output_dir: str = "outputs/radar"


@dataclass
class RadarFrame:
    sensor_id: str
    frame_idx: int
    radar_cube: np.ndarray  # shape: (num_chirps, num_antennas, num_samples)


class RadarSimulator:
    """
    Sionna RT FMCW radar simulator.

    This implementation computes Sionna RT paths once per frame and then
    synthesizes the FMCW dechirped beat signal from path delays, gains, and
    Doppler evolution. The saved output remains a single ``radar_cube`` with
    shape ``[num_chirps, num_antennas, num_samples]``.
    """

    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        self.path_solver = PathSolver()

        self.output_dir = Path(cfg.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _synthesize_fmcw_cube(self, paths, num_antennas: int) -> np.ndarray:
        """
        Convert Sionna RT paths into an FMCW dechirped beat signal.

        ``paths.cir`` returns baseband path coefficients evolved over chirps
        using Sionna's path Doppler model. The fast-time FMCW beat term is then
        applied from the path delay and chirp slope.
        """
        chirp_duration = float(self.cfg.t_pri)
        adc_sample_rate = float(self.cfg.num_samples) / chirp_duration
        slope = float(self.cfg.bandwidth) / chirp_duration

        path_coeffs, path_delays = paths.cir(
            sampling_frequency=1.0 / self.cfg.t_pri,
            num_time_steps=self.cfg.num_chirps,
            normalize_delays=False,
            out_type="numpy",
        )

        path_coeffs = np.asarray(path_coeffs)
        path_delays = np.asarray(path_delays)

        if path_delays.shape[-1] == 0 or path_coeffs.shape[-2] == 0:
            return np.zeros(
                (self.cfg.num_chirps, num_antennas, self.cfg.num_samples),
                dtype=np.complex64,
            )

        # Expected non-synthetic-array shape:
        # coeffs: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_chirps]
        # delays: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        # Some Sionna configurations may use synthetic arrays, in which case
        # delays omit antenna dimensions. Broadcast them to the RX antennas.
        coeffs = path_coeffs[0, :, 0, 0, :, :]
        if path_delays.ndim == 5:
            delays = path_delays[0, :, 0, 0, :]
        elif path_delays.ndim == 3:
            delays = np.broadcast_to(path_delays[0, 0, :], coeffs.shape[:2])
        else:
            raise ValueError(f"[Radar] Unexpected delay shape: {path_delays.shape}")

        valid = np.isfinite(delays) & (delays >= 0.0)
        coeffs = np.where(valid[:, :, None], coeffs, 0.0)
        delays = np.where(valid, delays, 0.0)

        t_fast = np.arange(self.cfg.num_samples, dtype=np.float64) / adc_sample_rate
        beat_phase = np.exp(
            1j * 2.0 * np.pi * slope * delays[:, :, None] * t_fast[None, None, :]
        )

        # [num_ant, num_paths, num_chirps, num_samples]
        beat_paths = coeffs[:, :, :, None] * beat_phase[:, :, None, :]
        # Sum over paths -> [num_ant, num_chirps, num_samples]
        radar_cube = np.sum(beat_paths, axis=1)
        # -> [num_chirps, num_ant, num_samples]
        radar_cube = np.transpose(radar_cube, (1, 0, 2))
        return radar_cube.astype(np.complex64, copy=False)

    def _add_awgn(self, radar_cube: np.ndarray, frame_idx: int) -> np.ndarray:
        if self.cfg.noise_snr_db is None:
            return radar_cube

        signal_power = float(np.mean(np.abs(radar_cube) ** 2))
        if not np.isfinite(signal_power) or signal_power <= 0.0:
            return radar_cube

        snr_linear = 10.0 ** (float(self.cfg.noise_snr_db) / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2.0)

        rng = np.random.default_rng(int(self.cfg.seed) + int(frame_idx))
        noise = noise_std * (
            rng.standard_normal(radar_cube.shape) + 1j * rng.standard_normal(radar_cube.shape)
        )
        return (radar_cube + noise).astype(np.complex64, copy=False)

    def simulate(self, scene, frame_idx: int) -> RadarFrame:
        """
        Generate one frame of radar data with shape
        ``[num_chirps, num_antennas, num_samples]``.
        """
        scene.frequency = self.cfg.carrier_frequency
        num_antennas = scene.rx_array.array_size

        tx = scene.get("veh_sens_tx")
        if tx is None:
            raise ValueError("[Radar] 'veh_sens_tx' not found in scene")

        base_pos = np.asarray(tx.position, dtype=float).reshape(3,)
        velocity = np.asarray(tx.velocity, dtype=float).reshape(3,)

        # Keep the frame-start position fixed and let Sionna evolve the phase
        # across chirps using Doppler.
        tx.position = base_pos.tolist()
        tx.velocity = velocity.tolist()

        paths = self.path_solver(
            scene=scene,
            los=True,
            samples_per_src=self.cfg.samples_per_src,
            max_num_paths_per_src=self.cfg.max_num_paths_per_src,
            max_depth=self.cfg.max_depth,
            specular_reflection=True,
            refraction=True,
            diffraction=False,
            seed=self.cfg.seed,
        )

        radar_cube = self._synthesize_fmcw_cube(paths, num_antennas)
        radar_cube = self._add_awgn(radar_cube, frame_idx)
        expected_shape = (self.cfg.num_chirps, num_antennas, self.cfg.num_samples)
        if radar_cube.shape != expected_shape:
            raise ValueError(
                "[Radar] Unexpected FMCW cube shape. "
                f"got={radar_cube.shape}, expected={expected_shape}"
            )

        return RadarFrame(
            sensor_id=self.cfg.sensor_id,
            frame_idx=frame_idx,
            radar_cube=radar_cube,
        )

    def save_frame(self, frame: RadarFrame, output_dir=None) -> Path:
        save_dir = Path(output_dir).resolve() if output_dir is not None else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"frame_{frame.frame_idx:06d}.npz"

        np.savez_compressed(
            save_path,
            radar_cube=frame.radar_cube,
        )

        return save_path
