import streamlit as st
import numpy as np
import pandas as pd
import os
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

# --- (0) Streamlit 頁面設定 ---
try:
    st.set_page_config(layout="wide", page_title="智慧空調 DEMO (含開門影響)")
except st.errors.StreamlitAPIException:
    pass

# --- (1) 修改後的 ACEnvWithDoors 環境定義 (與 Colab 訓練時一致) ---
class ACEnvWithDoors(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 initial_room_temp=28.0,
                 user_target_temp=24.0,
                 num_open_doors=0,
                 outside_temp=32.0,
                 render_mode=None,
                 initial_room_temp_range=(25.0, 35.0),
                 user_target_temp_range=(22.0, 26.0),
                 outside_temp_range=(10.0, 45.0), # 確保 observation_space 使用
                 initial_open_doors_min=0,
                 initial_open_doors_max=2,
                 max_open_doors=5,
                 door_change_prob=0.0, # DEMO中手動控制，設為0
                 door_impact_degrees=5.0 # 開門影響設為5度
                 ):
        super(ACEnvWithDoors, self).__init__()
        self.render_mode = render_mode

        self.initial_room_temp_val = float(initial_room_temp)
        self.user_target_temp_val = float(user_target_temp)
        self.num_open_doors_val = int(num_open_doors)
        self.outside_temp_val = float(outside_temp)

        self.initial_room_temp_range = initial_room_temp_range
        self.user_target_temp_range = user_target_temp_range
        self.outside_temp_range = outside_temp_range
        self.initial_open_doors_min = initial_open_doors_min
        self.initial_open_doors_max = initial_open_doors_max

        self.max_open_doors = max_open_doors
        self.door_change_prob = door_change_prob
        self.door_impact_degrees = door_impact_degrees

        self.min_room_temp = 10.0
        self.max_room_temp = 40.0
        self.min_user_target_temp_limit = 18.0
        self.max_user_target_temp_limit = 30.0
        self.min_ac_set_temp = 16.0
        self.max_ac_set_temp = 30.0
        self.min_fan_speed = 0.0
        self.max_fan_speed = 1.0

        self.heat_per_person_watt = 0
        self.room_thermal_mass_J_per_C = 1000 * 100
        self.wall_conductivity_W_per_C = 20
        self.ac_effectiveness_watt_per_deltaT_per_fan = 200
        self.ac_power_consumption_base_watt = 50
        self.ac_power_fan_coeff_watt = 200
        self.ac_power_cooling_coeff_watt = 500
        self.time_step_seconds = 5 * 60

        self.comfort_tolerance_C = 0.5
        self.comfort_reward_bonus = 1.0
        self.comfort_penalty_factor = 0.1
        self.energy_penalty_factor = 0.0001

        self.max_episode_steps = 1000
        self.current_step_count_in_episode = 0

        self.current_room_temp = self.initial_room_temp_val
        self.user_target_temp = self.user_target_temp_val
        self.num_open_doors = self.num_open_doors_val
        self.outside_temp = self.outside_temp_val
        self.previous_num_open_doors = self.num_open_doors

        self.observation_space = spaces.Box(
            low=np.array([self.min_room_temp, self.min_user_target_temp_limit, 0, self.outside_temp_range[0]], dtype=np.float32),
            high=np.array([self.max_room_temp, self.max_user_target_temp_limit, self.max_open_doors, self.outside_temp_range[1]], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([self.min_ac_set_temp, self.min_fan_speed], dtype=np.float32),
            high=np.array([self.max_ac_set_temp, self.max_fan_speed], dtype=np.float32),
            dtype=np.float32
        )
        if self.render_mode == "human":
            print("ACEnvWithDoors human render mode initialized.")

    def _get_obs(self):
        return np.array([self.current_room_temp, self.user_target_temp, float(self.num_open_doors), self.outside_temp], dtype=np.float32)

    def _get_info(self):
        return {
            "current_room_temp": self.current_room_temp,
            "user_target_temp": self.user_target_temp,
            "num_open_doors": self.num_open_doors,
            "outside_temp": self.outside_temp,
            "current_step": self.current_step_count_in_episode
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            self.current_room_temp = float(options.get("initial_room_temp", self.initial_room_temp_val))
            self.user_target_temp = float(options.get("user_target_temp", self.user_target_temp_val))
            self.num_open_doors = int(options.get("num_open_doors", self.num_open_doors_val))
            self.outside_temp = float(options.get("outside_temp", self.outside_temp_val))
        else:
            self.current_room_temp = self.initial_room_temp_val
            self.user_target_temp = self.user_target_temp_val
            self.num_open_doors = self.num_open_doors_val
            self.outside_temp = self.outside_temp_val

        self.previous_num_open_doors = self.num_open_doors
        self.current_step_count_in_episode = 0
        if self.render_mode == "human":
            self._render_frame(self._get_info(), is_reset=True)
        return self._get_obs(), self._get_info()

    def step(self, action):
        if not (isinstance(action, (np.ndarray, list)) and len(action) == 2):
            action = np.array([self.user_target_temp, 0.1])

        ac_set_temp = np.clip(action[0], self.min_ac_set_temp, self.max_ac_set_temp)
        fan_speed = np.clip(action[1], self.min_fan_speed, self.max_fan_speed)

        if self.num_open_doors > self.previous_num_open_doors:
            newly_opened_count = self.num_open_doors - self.previous_num_open_doors
            if self.outside_temp > self.current_room_temp:
                self.current_room_temp += self.door_impact_degrees * newly_opened_count
            elif self.outside_temp < self.current_room_temp:
                self.current_room_temp -= self.door_impact_degrees * newly_opened_count
            self.current_room_temp = np.clip(self.current_room_temp, self.min_room_temp, self.max_room_temp)

        self.previous_num_open_doors = self.num_open_doors

        heat_from_people_W = 0
        cooling_potential_delta_T = self.current_room_temp - ac_set_temp
        actual_cooling_power_W = 0.0
        if cooling_potential_delta_T > 0 and fan_speed > 0.01:
            actual_cooling_power_W = cooling_potential_delta_T * fan_speed * self.ac_effectiveness_watt_per_deltaT_per_fan
        heat_exchange_with_outside_W = (self.current_room_temp - self.outside_temp) * self.wall_conductivity_W_per_C
        net_heat_change_rate_W = heat_from_people_W - actual_cooling_power_W - heat_exchange_with_outside_W
        delta_temp_C = (net_heat_change_rate_W * self.time_step_seconds) / self.room_thermal_mass_J_per_C
        self.current_room_temp += delta_temp_C
        self.current_room_temp = np.clip(self.current_room_temp, self.min_room_temp, self.max_room_temp)

        energy_consumption_this_step_J = (
            self.ac_power_consumption_base_watt * (1 if fan_speed > 0.01 else 0) +
            self.ac_power_fan_coeff_watt * fan_speed +
            self.ac_power_cooling_coeff_watt * (actual_cooling_power_W / (self.ac_effectiveness_watt_per_deltaT_per_fan * (self.max_room_temp - self.min_ac_set_temp) + 1e-6) if actual_cooling_power_W > 0 else 0)
        ) * self.time_step_seconds
        energy_consumption_this_step_J = max(0, energy_consumption_this_step_J)

        temp_diff_comfort = abs(self.current_room_temp - self.user_target_temp)
        reward_comfort = 0.0
        if temp_diff_comfort <= self.comfort_tolerance_C:
            reward_comfort = self.comfort_reward_bonus
        else:
            reward_comfort = - (temp_diff_comfort - self.comfort_tolerance_C)**2 * self.comfort_penalty_factor
        reward_energy = - energy_consumption_this_step_J * self.energy_penalty_factor
        current_step_reward = reward_comfort + reward_energy

        self.current_step_count_in_episode += 1
        terminated = False
        truncated = self.current_step_count_in_episode >= self.max_episode_steps

        info = self._get_info()
        info['ac_set_temp_action'] = ac_set_temp
        info['fan_speed_action'] = fan_speed
        info['actual_cooling_power_W'] = actual_cooling_power_W
        info['energy_consumption_kJ'] = energy_consumption_this_step_J / 1000.0
        info['current_step_reward'] = current_step_reward
        info['reward_comfort'] = reward_comfort
        info['reward_energy'] = reward_energy

        if self.render_mode == "human":
            self._render_frame(info)
        return self._get_obs(), current_step_reward, terminated, truncated, info

    def _render_frame(self, info_dict, is_reset=False):
        if self.render_mode == "human":
            print(f"Step: {info_dict.get('current_step', 0):<4} | "
                  f"RoomTemp: {info_dict.get('current_room_temp'):>5.2f}°C | "
                  f"Target: {info_dict.get('user_target_temp'):>5.2f}°C | "
                  f"Doors: {info_dict.get('num_open_doors'):>1} | "
                  f"Outside: {info_dict.get('outside_temp'):>5.2f}°C | ", end="")
            if not is_reset:
                  print(f"AC_Set: {info_dict.get('ac_set_temp_action', 'N/A'):>5.2f}°C | "
                        f"Fan: {info_dict.get('fan_speed_action', 'N/A'):>4.2f} | "
                        f"Energy_kJ: {info_dict.get('energy_consumption_kJ', 0.0):>7.3f} | "
                        f"R_Comf: {info_dict.get('reward_comfort', 0.0):>6.2f} | "
                        f"R_Ener: {info_dict.get('reward_energy', 0.0):>7.3f} | "
                        f"R_Total: {info_dict.get('total_reward_step', 0.0):>7.3f}")
            else:
                print("Environment Reset.")

    def close(self):
        pass
# --- ACEnvWithDoors 定義結束 ---

# --- (2) 修改後的傳統控制器邏輯 ---
def traditional_controller(current_room_temp,
                           user_target_temp,
                           current_ac_set_temp,
                           env_min_ac_set_temp=16.0,
                           env_max_ac_set_temp=30.0,
                           temp_adjust_unit=1.0):

    delta_T = current_room_temp - user_target_temp
    next_ac_set_temp = float(current_ac_set_temp)

    fan_levels_map = {1: 0.1, 2: 0.3, 3: 0.6, 4: 0.8, 5: 1.0}
    action_fan_level = 1

    if delta_T > 1.5:
        desired_set_temp = max(env_min_ac_set_temp, next_ac_set_temp - temp_adjust_unit)
        next_ac_set_temp = round(desired_set_temp / temp_adjust_unit) * temp_adjust_unit
        action_fan_level = 5
    elif delta_T > 0.5:
        desired_set_temp = max(env_min_ac_set_temp, next_ac_set_temp - temp_adjust_unit)
        if desired_set_temp < user_target_temp - 0.5 :
             desired_set_temp = user_target_temp - temp_adjust_unit
        next_ac_set_temp = round(desired_set_temp / temp_adjust_unit) * temp_adjust_unit
        action_fan_level = 3
    elif delta_T < -1.5:
        desired_set_temp = min(env_max_ac_set_temp, user_target_temp, next_ac_set_temp + temp_adjust_unit)
        next_ac_set_temp = round(desired_set_temp / temp_adjust_unit) * temp_adjust_unit
        action_fan_level = 1
    elif delta_T < -0.5:
        desired_set_temp = min(env_max_ac_set_temp, user_target_temp, next_ac_set_temp + temp_adjust_unit)
        next_ac_set_temp = round(desired_set_temp / temp_adjust_unit) * temp_adjust_unit
        action_fan_level = 1
    else:
        next_ac_set_temp = round(user_target_temp / temp_adjust_unit) * temp_adjust_unit
        action_fan_level = 2

    next_ac_set_temp = np.clip(next_ac_set_temp, env_min_ac_set_temp, env_max_ac_set_temp)
    next_ac_set_temp = round(next_ac_set_temp / temp_adjust_unit) * temp_adjust_unit
    action_fan_speed = fan_levels_map.get(action_fan_level, 0.1)

    return np.array([next_ac_set_temp, action_fan_speed], dtype=np.float32)
# --- 傳統控制器定義結束 ---

# --- (3) SAC 模型載入邏輯 ---
MODEL_PATH = r"final_v10.zip" # 使用原始字串確保路徑正確

@st.cache_resource
def load_sac_model_cached(path):
    if not os.path.exists(path):
        st.error(f"SAC 模型檔案未找到於: {path}。請檢查路徑。")
        return None
    try:
        model = SAC.load(path)
        st.success(f"SAC 模型從 {path} 載入成功！")
        return model
    except Exception as e:
        st.error(f"載入 SAC 模型 '{path}' 失敗: {e}")
        return None

if 'sac_model_loaded_final' not in st.session_state:
    st.session_state.sac_model_loaded_final = load_sac_model_cached(MODEL_PATH)

# --- (4) Session State 初始化 ---
if 'running_final' not in st.session_state: st.session_state.running_final = False
if 'initialized_final' not in st.session_state: st.session_state.initialized_final = False
if 'current_time_step_final' not in st.session_state: st.session_state.current_time_step_final = 0

DEFAULT_HISTORY_DF_FINAL = pd.DataFrame(columns=[
    '時間步',
    'SAC 室內溫度 (°C)', 'SAC 設定溫度 (°C)', 'SAC 風扇強度', 'SAC 開門數', 'SAC 累積能耗 (kJ)', 'SAC 當步獎勵', 'SAC 累積獎勵',
    '傳統控制 室內溫度 (°C)', '傳統控制 設定溫度 (°C)', '傳統控制 風扇強度', '傳統控制 開門數', '傳統控制 累積能耗 (kJ)',
    '使用者目標溫度 (°C)', '室外溫度 (°C)'
])
if 'history_final' not in st.session_state:
    st.session_state.history_final = DEFAULT_HISTORY_DF_FINAL.copy()

default_initial_room_temp = 28.0
default_user_target_temp = 24.0
default_num_open_doors = 0
default_outside_temp = 32.0
default_sim_steps = 50
default_ui_update_delay = 0.3

# --- 輔助函式 ---
def initialize_final_simulation():
    initial_room_temp = st.session_state.get("slider_initial_room_temp_final", default_initial_room_temp)
    user_target_temp = st.session_state.get("slider_user_target_temp_final", default_user_target_temp)
    num_open_doors = st.session_state.get("slider_num_open_doors_final", default_num_open_doors)
    outside_temp = st.session_state.get("slider_outside_temp_final", default_outside_temp)
    
    st.session_state.current_time_step_final = 0
    st.session_state.current_user_target_temp_final = user_target_temp
    st.session_state.current_outside_temp_final = outside_temp
    st.session_state.current_num_open_doors_final = num_open_doors

    env_init_params = {
        "initial_room_temp_range": (10.0, 40.0),
        "user_target_temp_range": (18.0, 30.0),
        "outside_temp_range": (10.0, 45.0),
        "initial_open_doors_min": 0,
        "initial_open_doors_max": 2,
        "max_open_doors": 5,
        "door_change_prob": 0.0,
        "door_impact_degrees": 5.0
    }

    env_options_sac = {
        "initial_room_temp": initial_room_temp, "user_target_temp": user_target_temp,
        "num_open_doors": num_open_doors, "outside_temp": outside_temp,
        "render_mode": None, **env_init_params
    }
    env_options_trad = {
        "initial_room_temp": initial_room_temp, "user_target_temp": user_target_temp,
        "num_open_doors": num_open_doors, "outside_temp": outside_temp,
        "render_mode": None, **env_init_params
    }

    st.session_state.sac_env_final = ACEnvWithDoors(**env_options_sac)
    sac_obs, _ = st.session_state.sac_env_final.reset(options=env_options_sac)
    st.session_state.sac_current_obs_final = sac_obs
    st.session_state.sac_current_room_temp_final = st.session_state.sac_env_final.current_room_temp
    st.session_state.sac_action_final = np.array([user_target_temp, 0.1])
    st.session_state.sac_cool_power_final = 0.0
    st.session_state.sac_step_energy_final = 0.0
    st.session_state.sac_total_energy_final = 0.0
    st.session_state.sac_step_reward_final = 0.0
    st.session_state.sac_cumulative_reward_final = 0.0
    st.session_state.sac_comfort_reward_final = 0.0
    st.session_state.sac_energy_reward_final = 0.0
    st.session_state.sac_num_open_doors_final = num_open_doors

    st.session_state.trad_env_final = ACEnvWithDoors(**env_options_trad)
    _, _ = st.session_state.trad_env_final.reset(options=env_options_trad)
    st.session_state.trad_current_room_temp_final = st.session_state.trad_env_final.current_room_temp
    st.session_state.trad_action_final = np.array([user_target_temp, 0.1])
    st.session_state.trad_cool_power_final = 0.0
    st.session_state.trad_step_energy_final = 0.0
    st.session_state.trad_total_energy_final = 0.0
    st.session_state.trad_num_open_doors_final = num_open_doors

    st.session_state.history_final = DEFAULT_HISTORY_DF_FINAL.copy()
    log_current_state_final()

    st.session_state.initialized_final = True
    st.session_state.running_final = False

def update_final_environment():
    current_ui_target_temp = st.session_state.current_user_target_temp_final
    current_ui_outside_temp = st.session_state.current_outside_temp_final
    current_ui_num_open_doors = st.session_state.current_num_open_doors_final

    # 更新SAC環境的內部狀態
    st.session_state.sac_env_final.user_target_temp = current_ui_target_temp
    st.session_state.sac_env_final.outside_temp = current_ui_outside_temp
    st.session_state.sac_env_final.num_open_doors = current_ui_num_open_doors

    sac_action_to_env = np.array([current_ui_target_temp, 0.1])
    if st.session_state.sac_model_loaded_final:
        try:
            current_obs_for_sac = st.session_state.sac_env_final._get_obs()
            raw_action, _ = st.session_state.sac_model_loaded_final.predict(current_obs_for_sac, deterministic=True)
            sac_action_to_env = np.array(raw_action).flatten()
        except Exception as e:
            st.error(f"SAC 模型預測時出錯: {e}")
            current_sac_ac_set_temp_for_trad_fallback = st.session_state.get('sac_action_final', np.array([current_ui_target_temp, 0.1]))[0]
            sac_action_to_env = traditional_controller(st.session_state.sac_current_room_temp_final, current_ui_target_temp, current_sac_ac_set_temp_for_trad_fallback)

    sac_obs_next, sac_reward_from_step, sac_terminated, sac_truncated, sac_info = st.session_state.sac_env_final.step(sac_action_to_env)
    if sac_terminated or sac_truncated:
        st.session_state.running_final = False
    st.session_state.sac_current_obs_final = sac_obs_next
    st.session_state.sac_current_room_temp_final = sac_info['current_room_temp']
    st.session_state.sac_action_final = np.array([sac_info['ac_set_temp_action'], sac_info['fan_speed_action']])
    st.session_state.sac_cool_power_final = sac_info['actual_cooling_power_W']
    st.session_state.sac_step_energy_final = sac_info['energy_consumption_kJ']
    st.session_state.sac_total_energy_final += sac_info['energy_consumption_kJ']
    st.session_state.sac_step_reward_final = sac_info.get('current_step_reward', sac_reward_from_step)
    st.session_state.sac_comfort_reward_final = sac_info.get('reward_comfort', 0.0)
    st.session_state.sac_energy_reward_final = sac_info.get('reward_energy', 0.0)
    st.session_state.sac_cumulative_reward_final += st.session_state.sac_step_reward_final
    st.session_state.sac_num_open_doors_final = sac_info['num_open_doors']


    # 更新傳統控制環境的內部狀態
    st.session_state.trad_env_final.user_target_temp = current_ui_target_temp
    st.session_state.trad_env_final.outside_temp = current_ui_outside_temp
    st.session_state.trad_env_final.num_open_doors = current_ui_num_open_doors

    current_trad_ac_set_temp = st.session_state.get('trad_action_final', np.array([current_ui_target_temp, 0.1]))[0]
    trad_action = traditional_controller(
        st.session_state.trad_current_room_temp_final, current_ui_target_temp, current_trad_ac_set_temp,
        env_min_ac_set_temp=st.session_state.trad_env_final.min_ac_set_temp,
        env_max_ac_set_temp=st.session_state.trad_env_final.max_ac_set_temp
    )
    _, _, trad_terminated, trad_truncated, trad_info = st.session_state.trad_env_final.step(trad_action)
    if (trad_terminated or trad_truncated) and st.session_state.running_final :
        st.session_state.running_final = False
    st.session_state.trad_current_room_temp_final = trad_info['current_room_temp']
    st.session_state.trad_action_final = np.array([trad_info['ac_set_temp_action'], trad_info['fan_speed_action']])
    st.session_state.trad_cool_power_final = trad_info['actual_cooling_power_W']
    st.session_state.trad_step_energy_final = trad_info['energy_consumption_kJ']
    st.session_state.trad_total_energy_final += trad_info['energy_consumption_kJ']
    st.session_state.trad_num_open_doors_final = trad_info['num_open_doors']


def log_current_state_final():
    new_data = pd.DataFrame([{
        '時間步': st.session_state.current_time_step_final,
        'SAC 室內溫度 (°C)': st.session_state.sac_current_room_temp_final,
        'SAC 設定溫度 (°C)': st.session_state.sac_action_final[0],
        'SAC 風扇強度': st.session_state.sac_action_final[1],
        'SAC 開門數': st.session_state.sac_num_open_doors_final,
        'SAC 累積能耗 (kJ)': st.session_state.sac_total_energy_final,
        'SAC 當步獎勵': st.session_state.sac_step_reward_final,
        'SAC 累積獎勵': st.session_state.sac_cumulative_reward_final,
        '傳統控制 室內溫度 (°C)': st.session_state.trad_current_room_temp_final,
        '傳統控制 設定溫度 (°C)': st.session_state.trad_action_final[0],
        '傳統控制 風扇強度': st.session_state.trad_action_final[1],
        '傳統控制 開門數': st.session_state.trad_num_open_doors_final,
        '傳統控制 累積能耗 (kJ)': st.session_state.trad_total_energy_final,
        '使用者目標溫度 (°C)': st.session_state.current_user_target_temp_final, # 從 session_state 獲取
        '室外溫度 (°C)': st.session_state.current_outside_temp_final, # 從 session_state 獲取
    }])
    st.session_state.history_final = pd.concat([st.session_state.history_final, new_data], ignore_index=True)

# --- Streamlit 介面 ---
st.title("❄️ 智慧空調 DEMO (含開門影響)")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ 控制面板")
    user_target_temp_final_input = st.slider(
        "您的期望室內溫度 (°C)", 18.0, 30.0,
        value=st.session_state.get('slider_user_target_temp_final', default_user_target_temp),
        step=0.5, key='slider_user_target_temp_final',
        on_change=lambda: setattr(st.session_state, 'current_user_target_temp_final', st.session_state.slider_user_target_temp_final) if 'initialized_final' in st.session_state and st.session_state.initialized_final else None
    )
    initial_room_temp_final_input = st.number_input(
        "初始室內溫度 (°C)", min_value=10.0, max_value=40.0,
        value=st.session_state.get('slider_initial_room_temp_final', default_initial_room_temp),
        step=0.1, key="slider_initial_room_temp_final"
    )
    outside_temp_final_input = st.slider(
        "室外溫度 (°C)", 10.0, 45.0,
        value=st.session_state.get('slider_outside_temp_final', default_outside_temp),
        step=0.1, key="slider_outside_temp_final",
        on_change=lambda: setattr(st.session_state, 'current_outside_temp_final', st.session_state.slider_outside_temp_final) if 'initialized_final' in st.session_state and st.session_state.initialized_final else None
    )
    num_open_doors_final_input = st.slider(
        "目前開門數 (0-5)", 0, 5,
        value=st.session_state.get('slider_num_open_doors_final', default_num_open_doors),
        step=1, key="slider_num_open_doors_final",
        on_change=lambda: setattr(st.session_state, 'current_num_open_doors_final', st.session_state.slider_num_open_doors_final) if 'initialized_final' in st.session_state and st.session_state.initialized_final else None
    )
    num_steps_final_input = st.number_input(
        "單次執行步數", min_value=1, max_value=200,
        value=st.session_state.get("num_steps_to_run_input_final", default_sim_steps),
        step=1, key="num_steps_to_run_input_final"
    )
    ui_update_delay_final = st.select_slider(
        "UI 更新延遲 (秒)",
        options=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
        value=st.session_state.get("ui_update_delay_final_slider", default_ui_update_delay),
        key="ui_update_delay_final_slider"
    )

    if st.button("初始化/重置模擬", key="reset_sim_final_doors"):
        initialize_final_simulation()
        st.success("模擬已初始化！")
        st.rerun()

    if not st.session_state.get('initialized_final', False):
        st.warning("請先初始化模擬。")
        if st.session_state.get('sac_model_loaded_final') is None:
             st.error(f"SAC 模型未能從 {MODEL_PATH} 載入。請確認模型路徑是否正確。")
        st.stop()

    if st.session_state.running_final:
        if st.button("暫停模擬", key="pause_sim_final_doors"):
            st.session_state.running_final = False
            st.session_state.steps_to_run_this_session_final = 0
            st.rerun()
    else:
        if st.button("啟動/繼續模擬", key="start_sim_final_doors"):
            if not st.session_state.get('initialized_final', False):
                 st.warning("請先初始化模擬後再啟動。")
            elif st.session_state.get('sac_model_loaded_final') is None:
                 st.error("SAC 模型未載入，無法啟動模擬。")
            else:
                st.session_state.running_final = True
                if st.session_state.get('steps_to_run_this_session_final', 0) <= 0:
                    st.session_state.steps_to_run_this_session_final = st.session_state.get("num_steps_to_run_input_final", default_sim_steps)
                st.rerun()
    
    st.markdown("---")
    st.subheader("ℹ️ 模擬資訊")
    st.write(f"目前總時間步: {st.session_state.current_time_step_final}")
    if st.session_state.get('initialized_final', False):
        st.write(f"SAC Env 內部步數: {st.session_state.sac_env_final.current_step_count_in_episode}")
        st.write(f"SAC Env 開門數: {st.session_state.sac_env_final.num_open_doors}")
        st.write(f"Trad Env 開門數: {st.session_state.trad_env_final.num_open_doors}")


    if st.session_state.running_final and st.session_state.get('steps_to_run_this_session_final', 0) > 0:
        st.success(f"模擬運行中... (本輪剩餘 {st.session_state.steps_to_run_this_session_final} 步)")
    elif st.session_state.get('initialized_final', False):
        st.info("模擬已暫停或未啟動。")

# 主要顯示區域
if st.session_state.get('initialized_final', False):
    current_ui_target_temp_for_logic = st.session_state.current_user_target_temp_final
    current_ui_outside_temp_for_logic = st.session_state.current_outside_temp_final
    current_ui_num_open_doors_for_logic = st.session_state.current_num_open_doors_final


    if st.session_state.running_final and st.session_state.get('steps_to_run_this_session_final', 0) > 0:
        sac_env_ok = st.session_state.sac_env_final.current_step_count_in_episode < st.session_state.sac_env_final.max_episode_steps
        trad_env_ok = st.session_state.trad_env_final.current_step_count_in_episode < st.session_state.trad_env_final.max_episode_steps

        if sac_env_ok and trad_env_ok:
            with st.spinner(f"執行第 {st.session_state.current_time_step_final + 1} 步..."):
                update_final_environment()
                st.session_state.current_time_step_final += 1
                log_current_state_final() # 移除參數，從session_state獲取
                st.session_state.steps_to_run_this_session_final -=1

                if st.session_state.steps_to_run_this_session_final <= 0:
                    st.session_state.running_final = False
                    st.success("請求的模擬步數執行完畢。")
        else:
            st.session_state.running_final = False
            st.warning("環境達到內部最大 episode 步數，模擬已停止。如需繼續，請重置模擬。")

    # --- 調整後的介面佈局 ---
    st.markdown("---")
    col_sac, col_label, col_trad = st.columns([2,1,2]) # 調整列的比例

    with col_sac:
        st.header("🤖 SAC")
    with col_label:
        st.markdown("<br><br>", unsafe_allow_html=True) # 增加一些垂直間距
    with col_trad:
        st.header("📜 傳統")
    st.markdown("---") # 分隔線

    # 目前室內溫度
    temp_sac = f"{st.session_state.get('sac_current_room_temp_final', default_initial_room_temp):.2f} °C"
    temp_trad = f"{st.session_state.get('trad_current_room_temp_final', default_initial_room_temp):.2f} °C"
    delta_sac = f"{st.session_state.get('sac_current_room_temp_final', default_initial_room_temp) - current_ui_target_temp_for_logic:.2f} °C"
    delta_trad = f"{st.session_state.get('trad_current_room_temp_final', default_initial_room_temp) - current_ui_target_temp_for_logic:.2f} °C"

    kpi_cols = st.columns([2,1,2])
    with kpi_cols[0]: st.metric(label="SAC", value=temp_sac, delta=delta_sac)
    with kpi_cols[1]: st.markdown(f"<p style='text-align: center; font-weight: bold; margin-top: 35px;'>目前室內溫度</p>", unsafe_allow_html=True)
    with kpi_cols[2]: st.metric(label="傳統", value=temp_trad, delta=delta_trad)
    st.markdown("---")

    # 冷氣設定溫度 (°C)
    set_temp_sac = f"{st.session_state.get('sac_action_final', [current_ui_target_temp_for_logic, 0.1])[0]:.1f}"
    set_temp_trad = f"{st.session_state.get('trad_action_final', [current_ui_target_temp_for_logic, 0.1])[0]:.1f}"
    kpi_cols = st.columns([2,1,2])
    with kpi_cols[0]: st.markdown(f"<p style='font-size: 24px; text-align: center;'>{set_temp_sac}</p>", unsafe_allow_html=True)
    with kpi_cols[1]: st.markdown(f"<p style='text-align: center; font-weight: bold; margin-top: 5px;'>冷氣設定溫度 (°C)</p>", unsafe_allow_html=True)
    with kpi_cols[2]: st.markdown(f"<p style='font-size: 24px; text-align: center;'>{set_temp_trad}</p>", unsafe_allow_html=True)
    st.markdown("---")

    # 風扇強度
    fan_sac = f"{st.session_state.get('sac_action_final', [current_ui_target_temp_for_logic, 0.1])[1]:.2f}"
    fan_trad = f"{st.session_state.get('trad_action_final', [current_ui_target_temp_for_logic, 0.1])[1]:.2f}"
    kpi_cols = st.columns([2,1,2])
    with kpi_cols[0]: st.markdown(f"<p style='font-size: 24px; text-align: center;'>{fan_sac}</p>", unsafe_allow_html=True)
    with kpi_cols[1]: st.markdown(f"<p style='text-align: center; font-weight: bold; margin-top: 5px;'>風扇強度</p>", unsafe_allow_html=True)
    with kpi_cols[2]: st.markdown(f"<p style='font-size: 24px; text-align: center;'>{fan_trad}</p>", unsafe_allow_html=True)
    st.markdown("---")

    # 即時製冷功率 (W)
    cool_power_sac = f"{st.session_state.get('sac_cool_power_final',0.0):.1f}"
    cool_power_trad = f"{st.session_state.get('trad_cool_power_final',0.0):.1f}"
    kpi_cols = st.columns([2,1,2])
    with kpi_cols[0]: st.info(f"SAC: {cool_power_sac} W")
    with kpi_cols[1]: st.markdown(f"<p style='text-align: center; font-weight: bold; margin-top: 5px;'>即時製冷功率</p>", unsafe_allow_html=True)
    with kpi_cols[2]: st.info(f"傳統: {cool_power_trad} W")
    st.markdown("---")

    # 本步能耗 (kJ)
    step_energy_sac = f"{st.session_state.get('sac_step_energy_final',0.0):.3f}"
    step_energy_trad = f"{st.session_state.get('trad_step_energy_final',0.0):.3f}"
    kpi_cols = st.columns([2,1,2])
    with kpi_cols[0]: st.warning(f"SAC: {step_energy_sac} kJ")
    with kpi_cols[1]: st.markdown(f"<p style='text-align: center; font-weight: bold; margin-top: 5px;'>本步能耗</p>", unsafe_allow_html=True)
    with kpi_cols[2]: st.warning(f"傳統: {step_energy_trad} kJ")
    st.markdown("---")

    # 累積總能耗 (kJ)
    total_energy_sac = f"{st.session_state.get('sac_total_energy_final',0.0):.2f}"
    total_energy_trad = f"{st.session_state.get('trad_total_energy_final',0.0):.2f}"
    kpi_cols = st.columns([2,1,2])
    with kpi_cols[0]: st.error(f"SAC: {total_energy_sac} kJ")
    with kpi_cols[1]: st.markdown(f"<p style='text-align: center; font-weight: bold; margin-top: 5px;'>累積總能耗</p>", unsafe_allow_html=True)
    with kpi_cols[2]: st.error(f"傳統: {total_energy_trad} kJ")
    st.markdown("---")

    # SAC 觀測值顯示
    st.subheader("🤖 SAC 模型觀測值")
    sac_obs_display = st.session_state.get('sac_current_obs_final', [0,0,0,0])
    obs_df = pd.DataFrame({
        "觀測參數": ["當前室溫 (°C)", "使用者目標 (°C)", "開門數", "室外溫度 (°C)"],
        "數值": [f"{sac_obs_display[0]:.2f}", f"{sac_obs_display[1]:.2f}", f"{sac_obs_display[2]:.0f}", f"{sac_obs_display[3]:.2f}"]
    })
    st.table(obs_df)
    st.markdown("---")


    st.header("📊 圖表比較")
    if not st.session_state.get('history_final', DEFAULT_HISTORY_DF_FINAL.copy()).empty:
        history_df_to_plot = st.session_state.history_final.copy()
        if not history_df_to_plot.empty and '時間步' in history_df_to_plot.columns and len(history_df_to_plot) > 0 :
            chart_data = history_df_to_plot.set_index('時間步')
            st.subheader("🌡️ 溫度變化曲線")
            st.line_chart(chart_data[['SAC 室內溫度 (°C)', '傳統控制 室內溫度 (°C)', '使用者目標溫度 (°C)']])
            
            st.subheader("🚪 開門數與室外溫度變化")
            st.line_chart(chart_data[['SAC 開門數', '室外溫度 (°C)']])

            st.subheader("⚡ 累積能耗曲線")
            st.line_chart(chart_data[['SAC 累積能耗 (kJ)', '傳統控制 累積能耗 (kJ)']])
        else:
            st.info("歷史數據格式不正確或為空，無法繪製主要圖表。")
    else:
        st.info("暫無歷史數據。請初始化並啟動模擬。")

    st.markdown("---")
    st.header("🏆 SAC 智能體獎勵分析")
    st.subheader("SAC 即時獎勵資訊:")
    st.text(f"當步總獎勵: {st.session_state.get('sac_step_reward_final', 0.0):.3f}")
    st.text(f"  - 舒適度獎勵: {st.session_state.get('sac_comfort_reward_final', 0.0):.3f}")
    st.text(f"  - 能耗懲罰部分: {st.session_state.get('sac_energy_reward_final', 0.0):.3f}")
    st.text(f"累積總獎勵: {st.session_state.get('sac_cumulative_reward_final', 0.0):.2f}")

    if not st.session_state.get('history_final', DEFAULT_HISTORY_DF_FINAL.copy()).empty:
        history_df_to_plot_reward = st.session_state.history_final.copy()
        if not history_df_to_plot_reward.empty and '時間步' in history_df_to_plot_reward.columns and \
           'SAC 當步獎勵' in history_df_to_plot_reward.columns and \
           'SAC 累積獎勵' in history_df_to_plot_reward.columns and \
           len(history_df_to_plot_reward) > 0:
            chart_data_reward = history_df_to_plot_reward.set_index('時間步')
            st.subheader("📈 SAC 獎勵曲線")
            st.line_chart(chart_data_reward[['SAC 當步獎勵', 'SAC 累積獎勵']])
        else:
            st.caption("SAC 獎勵圖表數據記錄中或不完整...")
    else:
        st.info("暫無歷史數據，無法繪製 SAC 獎勵圖表。")
else:
    st.info("應用程式初始化中或等待初始化...")

# --- 自動刷新 ---
if st.session_state.get('running_final', False) and st.session_state.get('steps_to_run_this_session_final', 0) > 0:
    sac_env = st.session_state.get('sac_env_final')
    trad_env = st.session_state.get('trad_env_final')
    if sac_env and trad_env:
        if sac_env.current_step_count_in_episode < sac_env.max_episode_steps and \
           trad_env.current_step_count_in_episode < trad_env.max_episode_steps:
            time.sleep(st.session_state.get("ui_update_delay_final_slider", default_ui_update_delay))
            st.rerun()
        else:
            if st.session_state.running_final:
                st.session_state.running_final = False
                st.warning("環境內部已達最大 episode 步數，模擬自動停止。")
                st.rerun()
    else:
        if st.session_state.running_final:
            st.session_state.running_final = False
            st.rerun()
elif st.session_state.get('running_final', False) and st.session_state.get('steps_to_run_this_session_final', 0) == 0:
    if st.session_state.running_final:
        st.session_state.running_final = False
        st.info("請求的步數已執行完畢，模擬已暫停。")
        st.rerun()
