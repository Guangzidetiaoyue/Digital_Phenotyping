# M4 外部流行病学数据

> 用途：RQ4 领先-滞后验证（数字信号 vs 传统流行病学指标）
> 获取方法：浏览器会话（AWS ELB 反爬，curl/服务器 403）；本地浏览器有 VPN

## 已获取数据（真实，浏览器亲自提取）

| 文件 | 内容 | 时间/粒度 | 来源 |
|---|---|---|---|
| `cdc_suicide_rate_1990_2018.csv` | 自杀死亡率（age-adjusted /10万） | 1990-2018 年度 | CDC `data.cdc.gov/d/9j2v-jamp` |
| `nsduh_mde_prevalence_2015_2024.csv` | MDE 患病率（18+ %）+ 方法学 regime | **2015-2024 完整** | SAMHSA NSDUH Detailed Tables（详见 `NSDUH_MDE_SOURCES.md`） |
| `cdc_monthly_suicide_2020_2023.csv` | 月度自杀死亡数（provisional） | 2020-2023 月度 | CDC `data.cdc.gov/d/9dzk-mvmi` |
| `mha_state_depression_2025.csv` | 52 州+全国成年抑郁患病率 | 2025 横截面 | Mental Health America |
| `NSDUH_MDE_SOURCES.md` | 各年份数据来源表 | — | 验证追溯 |

### 关键数字速览
- **CDC 自杀率**：1999 年 10.5 → 2018 年 14.8（持续上升）
- **NSDUH MDE（完整 10 年）**：2015=6.7% → 2019=7.8% → 2020=8.4% → 2022=8.8% 峰值 → 2024=8.2%
  - ⚠️ **2021 方法学断裂**：2015-2020（pre-2021）与 2021+（post-2021）不可直接连续
- **MHA 州级抑郁**：National 22.95%；州范围 19.69%（NJ）– 30.16%（Oregon）

## 需手动下载的补充数据（用户操作）

> 机构网站反爬 + zip 格式，浏览器无法自动解压，需人工下载后放至 `data/external/raw/`

### 1. SAMHSA 988 求助热线（月度，2021-2024）
- **来源**：https://www.samhsa.gov/mental-health/988/performance-metrics（交互仪表盘）
- **说明**：无直接 CSV，需从月度报告 PDF/仪表盘整理
- **建议变量**：calls, texts, chats, total_contacts, answer_rate
- **语义**：= 危机求助信号（非抑郁患病率）

### 2. CDC WONDER 自杀（月度/年度，更完整）
- **来源**：https://wonder.cdc.gov/（Underlying Cause of Death，ICD-10 X60-X84）
- **说明**：需交互查询导出；建议扩展 suicide_rate 至 2019-2023

### 3. BRFSS 心理健康天数（年度，2001-2023）
- **来源**：https://www.cdc.gov/brfss/（Annual Survey Data，XPT 格式）
- **⚠️ 关键**：需**变量可用性审计** + **复杂抽样加权**（不能用简单 mean）

## 与 M4 的衔接
- **RQ4a（时间外效）**：CDC 自杀率/NSDUH MDE/988（跨相关/Granger/滞后回归）
- **RQ4b（空间外效）**：MHA 州级抑郁 vs Reddit 州级信号
- ⚠️ 人群差异：外部=全国人口，我们=Reddit 用户 → 定性对照为主

## Processed ���ݣ�ͳһ��ʽ��s40 �ܵ����ɣ�

| �ļ� | ���� | ���� |
|---|---|---|
| `processed/epidemiology_annual.csv` | ���ָ�꣨suicide_rate + MDE_prevalence�� | 45 |
| `processed/epidemiology_monthly.csv` | �¶���ɱ���� | 45 |
| `processed/epidemiology_state.csv` | �ݼ����������� | 52 |
| `processed/variable_dictionary.csv` | ����Ԫ���ݣ���Դ/URL/ʱ��/����ѧ���ѣ� | 4 |

��ʽ��`year,indicator,value,unit,source,population,geography`����ֱ�����ڿ����/Granger/VAR��
