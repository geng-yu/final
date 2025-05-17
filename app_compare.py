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
    st.set_page_config(layout="wide", page_title="智慧空調 DEMO (獎勵移至底部)")
except st.errors.StreamlitAPIException:
    pass

# --- (1) 您原始的 ACEnv 環境定義 ---
# <<<< 確保這裡的 ACEnv 與您原始 app_compare.py 中的完全一致 >>>>
class ACEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 30}
    def __init__(self, initial_room_temp=28.0, user_target_temp=24.0, num_people=1, outside_temp=32.0, render_mode=None, use_legacy_obs_shape=False):
        super(ACEnv, self).__init__()
        self.render_mode = render_mode
        self.use_legacy_obs_shape = use_legacy_obs_shape
        self.min_room_temp = 10.0
        self.max_room_temp = 40.0
        self.min_user_target_temp_limit = 18.0
        self.max_user_target_temp_limit = 30.0
        self.min_ac_set_temp = 16.0
        self.max_ac_set_temp = 30.0
        self.min_fan_speed = 0.0
        self.max_fan_speed = 1.0
        self.min_people_limit = 0
        self.max_people_limit = 10
        self.initial_room_temp_val = float(initial_room_temp)
        self.user_target_temp_val = float(user_target_temp)
        self.num_people_val = int(num_people)
        self.outside_temp_val = float(outside_temp)
        self.heat_per_person_watt = 100
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

        self.current_room_temp = self.initial_room_temp_val
        self.user_target_temp = self.user_target_temp_val
        self.num_people = self.num_people_val
        self.current_step_count_in_episode = 0

        if self.use_legacy_obs_shape:
            self.observation_space = spaces.Box(
                low=np.array([self.min_room_temp, self.min_user_target_temp_limit], dtype=np.float32),
                high=np.array([self.max_room_temp, self.max_user_target_temp_limit], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([self.min_room_temp, self.min_user_target_temp_limit, self.min_people_limit], dtype=np.float32),
                high=np.array([self.max_room_temp, self.max_user_target_temp_limit, self.max_people_limit], dtype=np.float32),
                dtype=np.float32
            )
        self.action_space = spaces.Box(
            low=np.array([self.min_ac_set_temp, self.min_fan_speed], dtype=np.float32),
            high=np.array([self.max_ac_set_temp, self.max_fan_speed], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_room_temp = float(options.get("initial_room_temp", self.initial_room_temp_val)) if options else self.initial_room_temp_val
        self.user_target_temp = float(options.get("user_target_temp", self.user_target_temp_val)) if options else self.user_target_temp_val
        self.num_people = int(options.get("num_people", self.num_people_val)) if options else self.num_people_val
        if options and "outside_temp" in options:
            self.outside_temp_val = float(options["outside_temp"])
        if options and "use_legacy_obs_shape" in options:
            self.use_legacy_obs_shape = options["use_legacy_obs_shape"]
        self.current_step_count_in_episode = 0
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        if self.use_legacy_obs_shape:
            return np.array([self.current_room_temp, self.user_target_temp], dtype=np.float32)
        else:
            return np.array([self.current_room_temp, self.user_target_temp, float(self.num_people)], dtype=np.float32)

    def _get_info(self):
        return {
            "current_room_temp": self.current_room_temp,
            "user_target_temp": self.user_target_temp,
            "num_people": self.num_people,
            "outside_temp": self.outside_temp_val,
            "current_step": self.current_step_count_in_episode
        }

    def step(self, action):
        if not (isinstance(action, (np.ndarray, list)) and len(action) == 2):
            action = np.array([self.user_target_temp, 0.1])

        ac_set_temp = np.clip(action[0], self.min_ac_set_temp, self.max_ac_set_temp)
        fan_speed = np.clip(action[1], self.min_fan_speed, self.max_fan_speed)

        heat_from_people_W = self.num_people * self.heat_per_person_watt
        cooling_potential_delta_T = self.current_room_temp - ac_set_temp
        actual_cooling_power_W = 0.0
        if cooling_potential_delta_T > 0 and fan_speed > 0.01:
            actual_cooling_power_W = cooling_potential_delta_T * fan_speed * self.ac_effectiveness_watt_per_deltaT_per_fan
        
        heat_exchange_with_outside_W = (self.current_room_temp - self.outside_temp_val) * self.wall_conductivity_W_per_C
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
        
        return self._get_obs(), current_step_reward, terminated, truncated, info

    def close(self):
        pass
# --- ACEnv 定義結束 ---

# --- (2) 傳統控制器邏輯 ---
def traditional_controller(current_room_temp, user_target_temp, env_min_ac_set_temp=16.0, env_max_ac_set_temp=30.0):
    delta_T = current_room_temp - user_target_temp
    action_ac_set_temp = user_target_temp
    action_fan_speed = 0.1
    if delta_T > 1.5:
        action_ac_set_temp = env_min_ac_set_temp
        action_fan_speed = 1.0
    elif delta_T > 0.5:
        action_ac_set_temp = max(env_min_ac_set_temp, user_target_temp - 1.0)
        action_fan_speed = 0.6
    elif delta_T >= -0.5: 
        action_ac_set_temp = user_target_temp
        action_fan_speed = 0.2
    else: 
        action_ac_set_temp = min(env_max_ac_set_temp, user_target_temp + 1.0)
        action_fan_speed = 0.05
    return np.array([action_ac_set_temp, action_fan_speed], dtype=np.float32)

# --- (3) SAC 模型載入邏輯 ---
MODEL_PATH = "C:/Users/User/final_v10.zip" 
USE_LEGACY_OBSERVATION_SHAPE_FOR_SAC = False 
DEFAULT_FAN_FOR_LEGACY_ACTION = 0.1

@st.cache_resource
def load_sac_model_cached(path):
    if not os.path.exists(path):
        st.error(f"SAC 模型檔案未找到於: {path}。")
        return None
    try:
        model = SAC.load(path)
        st.success(f"SAC 模型從 {path} 載入成功！")
        return model
    except Exception as e:
        st.error(f"載入 SAC 模型失敗: {e}")
        return None

if 'sac_model_loaded_final' not in st.session_state:
    st.session_state.sac_model_loaded_final = load_sac_model_cached(MODEL_PATH)

# --- (4) Session State 初始化 ---
if 'running_final' not in st.session_state: st.session_state.running_final = False
if 'initialized_final' not in st.session_state: st.session_state.initialized_final = False
if 'current_time_step_final' not in st.session_state: st.session_state.current_time_step_final = 0

DEFAULT_HISTORY_DF_FINAL = pd.DataFrame(columns=[
    '時間步',
    'SAC 室內溫度 (°C)', 'SAC 設定溫度 (°C)', 'SAC 風扇強度', 'SAC 累積能耗 (kJ)', 'SAC 當步獎勵', 'SAC 累積獎勵',
    '傳統控制 室內溫度 (°C)', '傳統控制 設定溫度 (°C)', '傳統控制 風扇強度', '傳統控制 累積能耗 (kJ)',
    '使用者目標溫度 (°C)'
])
if 'history_final' not in st.session_state:
    st.session_state.history_final = DEFAULT_HISTORY_DF_FINAL.copy()

default_initial_room_temp = 28.0
default_user_target_temp = 24.0
default_num_people = 2
default_outside_temp = 32.0
default_sim_steps = 50
default_ui_update_delay = 0.3

# --- 輔助函式 ---
def initialize_final_simulation():
    initial_room_temp = st.session_state.get("slider_initial_room_temp_final", default_initial_room_temp)
    user_target_temp = st.session_state.get("slider_user_target_temp_final", default_user_target_temp)
    num_people = st.session_state.get("slider_num_people_final", default_num_people)
    outside_temp = st.session_state.get("slider_outside_temp_final", default_outside_temp)
    
    st.session_state.current_time_step_final = 0
    st.session_state.current_user_target_temp_final = user_target_temp

    env_options = {
        "initial_room_temp": initial_room_temp,
        "user_target_temp": user_target_temp,
        "num_people": num_people,
        "outside_temp": outside_temp,
        "use_legacy_obs_shape": USE_LEGACY_OBSERVATION_SHAPE_FOR_SAC
    }

    st.session_state.sac_env_final = ACEnv(**env_options)
    sac_obs, _ = st.session_state.sac_env_final.reset(options=env_options)
    st.session_state.sac_current_obs_final = sac_obs
    st.session_state.sac_current_room_temp_final = st.session_state.sac_env_final.current_room_temp
    st.session_state.sac_action_final = np.array([user_target_temp, DEFAULT_FAN_FOR_LEGACY_ACTION if USE_LEGACY_OBSERVATION_SHAPE_FOR_SAC else 0.1])
    st.session_state.sac_cool_power_final = 0.0
    st.session_state.sac_step_energy_final = 0.0
    st.session_state.sac_total_energy_final = 0.0
    st.session_state.sac_step_reward_final = 0.0
    st.session_state.sac_cumulative_reward_final = 0.0
    st.session_state.sac_comfort_reward_final = 0.0
    st.session_state.sac_energy_reward_final = 0.0

    st.session_state.trad_env_final = ACEnv(**env_options)
    trad_obs, _ = st.session_state.trad_env_final.reset(options=env_options)
    st.session_state.trad_current_room_temp_final = st.session_state.trad_env_final.current_room_temp
    st.session_state.trad_action_final = np.array([user_target_temp, 0.1])
    st.session_state.trad_cool_power_final = 0.0
    st.session_state.trad_step_energy_final = 0.0
    st.session_state.trad_total_energy_final = 0.0
    
    st.session_state.history_final = DEFAULT_HISTORY_DF_FINAL.copy()
    log_current_state_final(st.session_state.sac_env_final.user_target_temp)

    st.session_state.initialized_final = True
    st.session_state.running_final = False

def update_final_environment():
    current_ui_target_temp = st.session_state.get("slider_user_target_temp_final", default_user_target_temp)
    sac_action_to_env = np.array([current_ui_target_temp, 0.1])
    if st.session_state.sac_model_loaded_final:
        try:
            current_obs = st.session_state.sac_current_obs_final
            if not isinstance(current_obs, np.ndarray): current_obs = np.array(current_obs, dtype=np.float32)
            raw_action, _ = st.session_state.sac_model_loaded_final.predict(current_obs, deterministic=True)
            processed_action = np.array(raw_action).flatten()
            if len(processed_action) == 1 and USE_LEGACY_OBSERVATION_SHAPE_FOR_SAC:
                sac_action_to_env = np.array([processed_action[0], DEFAULT_FAN_FOR_LEGACY_ACTION])
            elif len(processed_action) == 2:
                sac_action_to_env = processed_action
            else:
                sac_action_to_env = st.session_state.sac_action_final
        except Exception as e:
            st.error(f"SAC 模型預測時出錯: {e}")
            sac_action_to_env = st.session_state.sac_action_final
    else:
        sac_action_to_env = traditional_controller(st.session_state.sac_current_room_temp_final, current_ui_target_temp -1)

    sac_obs, sac_reward_from_step, _, sac_truncated, sac_info = st.session_state.sac_env_final.step(sac_action_to_env)
    
    if sac_truncated:
        st.session_state.running_final = False
        st.warning("SAC 環境達到最大模擬步數，模擬已停止。")
    st.session_state.sac_current_obs_final = sac_obs
    st.session_state.sac_current_room_temp_final = sac_info['current_room_temp']
    st.session_state.sac_action_final = np.array([sac_info['ac_set_temp_action'], sac_info['fan_speed_action']])
    st.session_state.sac_cool_power_final = sac_info['actual_cooling_power_W']
    st.session_state.sac_step_energy_final = sac_info['energy_consumption_kJ']
    st.session_state.sac_total_energy_final += sac_info['energy_consumption_kJ']
    st.session_state.sac_step_reward_final = sac_info.get('current_step_reward', sac_reward_from_step)
    st.session_state.sac_comfort_reward_final = sac_info.get('reward_comfort', 0.0)
    st.session_state.sac_energy_reward_final = sac_info.get('reward_energy', 0.0)
    st.session_state.sac_cumulative_reward_final += st.session_state.sac_step_reward_final

    trad_action = traditional_controller(st.session_state.trad_current_room_temp_final, current_ui_target_temp)
    _, _, _, trad_truncated, trad_info = st.session_state.trad_env_final.step(trad_action)
    if trad_truncated and st.session_state.running_final:
        st.session_state.running_final = False
        st.warning("傳統控制環境達到最大模擬步數，模擬已停止。")
    st.session_state.trad_current_room_temp_final = trad_info['current_room_temp']
    st.session_state.trad_action_final = np.array([trad_info['ac_set_temp_action'], trad_info['fan_speed_action']])
    st.session_state.trad_cool_power_final = trad_info['actual_cooling_power_W']
    st.session_state.trad_step_energy_final = trad_info['energy_consumption_kJ']
    st.session_state.trad_total_energy_final += trad_info['energy_consumption_kJ']

def log_current_state_final(user_target_temp_for_log):
    new_data = pd.DataFrame([{
        '時間步': st.session_state.current_time_step_final,
        'SAC 室內溫度 (°C)': st.session_state.sac_current_room_temp_final,
        'SAC 設定溫度 (°C)': st.session_state.sac_action_final[0],
        'SAC 風扇強度': st.session_state.sac_action_final[1],
        'SAC 累積能耗 (kJ)': st.session_state.sac_total_energy_final,
        'SAC 當步獎勵': st.session_state.sac_step_reward_final,
        'SAC 累積獎勵': st.session_state.sac_cumulative_reward_final,
        '傳統控制 室內溫度 (°C)': st.session_state.trad_current_room_temp_final,
        '傳統控制 設定溫度 (°C)': st.session_state.trad_action_final[0],
        '傳統控制 風扇強度': st.session_state.trad_action_final[1],
        '傳統控制 累積能耗 (kJ)': st.session_state.trad_total_energy_final,
        '使用者目標溫度 (°C)': user_target_temp_for_log
    }])
    st.session_state.history_final = pd.concat([st.session_state.history_final, new_data], ignore_index=True)

# --- Streamlit 介面 ---
st.title("❄️ 智慧空調 DEMO (獎勵移至底部)")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ 控制面板")
    user_target_temp_final_input = st.slider(
        "您的期望室內溫度 (°C)", 18.0, 30.0,
        value=st.session_state.get('slider_user_target_temp_final', default_user_target_temp),
        step=0.5, key='slider_user_target_temp_final'
    )
    initial_room_temp_final_input = st.number_input(
        "初始室內溫度 (°C)", 15.0, 40.0,
        value=st.session_state.get('slider_initial_room_temp_final', default_initial_room_temp),
        step=0.1, key="slider_initial_room_temp_final"
    )
    num_people_final_input = st.slider(
        "室內人數", 0, 10,
        value=st.session_state.get('slider_num_people_final', default_num_people),
        step=1, key="slider_num_people_final"
    )
    outside_temp_final_input = st.slider(
        "室外溫度 (°C)", 20.0, 45.0,
        value=st.session_state.get('slider_outside_temp_final', default_outside_temp),
        step=0.1, key="slider_outside_temp_final"
    )
    num_steps_final_input = st.number_input(
        "執行次數", min_value=10, max_value=1000,
        value=st.session_state.get("num_steps_to_run_input_final", default_sim_steps),
        step=1, key="num_steps_to_run_input_final"
    )
    ui_update_delay_final = st.select_slider(
        "UI 更新延遲 (秒)",
        options=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
        value=st.session_state.get("ui_update_delay_final_slider", default_ui_update_delay),
        key="ui_update_delay_final_slider"
    )

    if st.button("初始化/重置模擬", key="reset_sim_final"):
        initialize_final_simulation()
        st.success("模擬已初始化！")
        st.rerun()

    if not st.session_state.get('initialized_final', False):
        st.warning("請先初始化模擬。")
        if st.session_state.get('sac_model_loaded_final') is None:
             st.error(f"SAC 模型未能從 {MODEL_PATH} 載入。")
        st.stop()

    if st.session_state.running_final:
        if st.button("暫停模擬", key="pause_sim_final"):
            st.session_state.running_final = False
            st.session_state.steps_to_run_this_session_final = 0
            st.rerun()
    else:
        if st.button("啟動模擬", key="start_sim_final"):
            if not st.session_state.get('initialized_final', False):
                 st.warning("請先初始化模擬後再啟動。")
            elif st.session_state.get('sac_model_loaded_final') is None:
                 st.error("SAC 模型未載入，無法啟動模擬。")
            else:
                st.session_state.running_final = True
                st.session_state.steps_to_run_this_session_final = st.session_state.get("num_steps_to_run_input_final", default_sim_steps)
                st.rerun()
    
    st.markdown("---")
    st.subheader("ℹ️ 模擬資訊")
    st.write(f"目前時間步: {st.session_state.current_time_step_final}")
    if st.session_state.running_final and st.session_state.get('steps_to_run_this_session_final',0) > 0:
        st.success(f"模擬運行中... (本輪剩餘 {st.session_state.steps_to_run_this_session_final} 步)")
    else:
        st.info("模擬已暫停或未啟動。")

if st.session_state.get('initialized_final', False):
    current_ui_target_temp_for_logic = st.session_state.get('slider_user_target_temp_final', default_user_target_temp)

    if st.session_state.running_final and st.session_state.get('steps_to_run_this_session_final', 0) > 0:
        if st.session_state.sac_env_final.current_step_count_in_episode < st.session_state.sac_env_final.max_episode_steps and \
           st.session_state.trad_env_final.current_step_count_in_episode < st.session_state.trad_env_final.max_episode_steps:
            
            with st.spinner(f"執行第 {st.session_state.current_time_step_final + 1} 步..."):
                update_final_environment()
                st.session_state.current_time_step_final += 1
                log_current_state_final(current_ui_target_temp_for_logic)
                st.session_state.steps_to_run_this_session_final -=1

                if st.session_state.steps_to_run_this_session_final <= 0:
                    st.session_state.running_final = False
                    st.success("請求的模擬步數執行完畢。")
        else:
            st.session_state.running_final = False
            st.warning("環境達到內部最大 episode 步數，模擬已停止。如需繼續，請重置模擬。")

    # --- 主要顯示區 (SAC 和傳統控制並排) ---
    col1_disp, col2_disp = st.columns(2)
    with col1_disp:
        st.header("🤖 SAC 智慧控制")
        st.metric("目前室內溫度", f"{st.session_state.get('sac_current_room_temp_final', default_initial_room_temp):.2f} °C",
                  delta=f"{st.session_state.get('sac_current_room_temp_final', default_initial_room_temp) - current_ui_target_temp_for_logic:.2f} °C vs 目標" if st.session_state.get('initialized_final', False) else None)
        sac_action_display_final = st.session_state.get('sac_action_final', [current_ui_target_temp_for_logic, 0.1])
        sac_act_df = pd.DataFrame({"參數": ["冷氣設定溫度 (°C)", "風扇強度"], "數值": [f"{sac_action_display_final[0]:.2f}", f"{sac_action_display_final[1]:.2f}"]})
        st.table(sac_act_df)
        st.info(f"即時製冷功率: {st.session_state.get('sac_cool_power_final',0.0):.1f} W")
        st.warning(f"本步能耗: {st.session_state.get('sac_step_energy_final',0.0):.3f} kJ")
        st.error(f"累積總能耗: {st.session_state.get('sac_total_energy_final',0.0):.2f} kJ")
        # SAC獎勵資訊移到下面

    with col2_disp:
        st.header("📜 傳統控制策略")
        st.metric("目前室內溫度", f"{st.session_state.get('trad_current_room_temp_final', default_initial_room_temp):.2f} °C",
                  delta=f"{st.session_state.get('trad_current_room_temp_final', default_initial_room_temp) - current_ui_target_temp_for_logic:.2f} °C vs 目標" if st.session_state.get('initialized_final', False) else None)
        trad_action_display_final = st.session_state.get('trad_action_final', [current_ui_target_temp_for_logic, 0.1])
        trad_act_df = pd.DataFrame({"參數": ["冷氣設定溫度 (°C)", "風扇強度"], "數值": [f"{trad_action_display_final[0]:.2f}", f"{trad_action_display_final[1]:.2f}"]})
        st.table(trad_act_df)
        st.info(f"即時製冷功率: {st.session_state.get('trad_cool_power_final',0.0):.1f} W")
        st.warning(f"本步能耗: {st.session_state.get('trad_step_energy_final',0.0):.3f} kJ")
        st.error(f"累積總能耗: {st.session_state.get('trad_total_energy_final',0.0):.2f} kJ")

    st.markdown("---")
    st.header("📊 圖表比較") # 主要圖表 (溫度、能耗)

    if not st.session_state.get('history_final', DEFAULT_HISTORY_DF_FINAL.copy()).empty:
        history_df_to_plot = st.session_state.history_final.copy()
        if not history_df_to_plot.empty and '時間步' in history_df_to_plot.columns and len(history_df_to_plot) > 0 :
            chart_data = history_df_to_plot.set_index('時間步')
            st.subheader("🌡️ 溫度變化曲線")
            st.line_chart(chart_data[['SAC 室內溫度 (°C)', '傳統控制 室內溫度 (°C)', '使用者目標溫度 (°C)']])
            
            st.subheader("⚡ 累積能耗曲線") # 將累積能耗圖也放在這裡
            st.line_chart(chart_data[['SAC 累積能耗 (kJ)', '傳統控制 累積能耗 (kJ)']])
        else:
            st.info("歷史數據格式不正確或為空，無法繪製主要圖表。")
    else:
        st.info("暫無歷史數據。請初始化並啟動模擬。")

    # --- SAC 獎勵資訊和圖表移至最下方 ---
    st.markdown("---") # 新的分隔線
    st.header("🏆 SAC 智能體獎勵分析")

    # 顯示 SAC 獎勵的文字資訊
    st.subheader("SAC 即時獎勵資訊:")
    st.text(f"當步總獎勵: {st.session_state.get('sac_step_reward_final', 0.0):.3f}")
    st.text(f"  - 舒適度獎勵: {st.session_state.get('sac_comfort_reward_final', 0.0):.3f}")
    st.text(f"  - 能耗懲罰部分: {st.session_state.get('sac_energy_reward_final', 0.0):.3f}") # 注意這裡可能是負值
    st.text(f"累積總獎勵: {st.session_state.get('sac_cumulative_reward_final', 0.0):.2f}")

    # 顯示 SAC 獎勵圖表
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

else: # initialized_final is False
    st.info("應用程式初始化中或等待初始化...")


# --- 自動刷新 ---
if st.session_state.get('running_final', False) and st.session_state.get('steps_to_run_this_session_final', 0) > 0:
    if st.session_state.sac_env_final.current_step_count_in_episode < st.session_state.sac_env_final.max_episode_steps and \
       st.session_state.trad_env_final.current_step_count_in_episode < st.session_state.trad_env_final.max_episode_steps:
        time.sleep(st.session_state.get("ui_update_delay_final_slider", default_ui_update_delay))
        st.rerun()
    else:
        if st.session_state.running_final:
            st.session_state.running_final = False
            st.warning("環境內部已達最大步數，自動刷新停止。")
            st.rerun()
elif st.session_state.get('running_final', False) and st.session_state.get('steps_to_run_this_session_final', 0) == 0:
    if st.session_state.running_final:
        st.session_state.running_final = False
        st.rerun()