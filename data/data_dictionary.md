# AIIMS Raipur Outpatient Flow Dataset: Data Dictionary & Grounding

## 1. Overview
This dataset contains **52,000 granular outpatient journey records** across **32 clinical OPD departments** at the All India Institute of Medical Sciences (AIIMS), Tatibandh, GE Road, Raipur, Chhattisgarh.

The dataset captures operational reality, patient demographic diversity, Chhattisgarh-specific socio-geographic dynamics, seasonal and post-holiday arrival surges, and realistic log-normal clinical consultation durations.

---

## 2. Feature Schema & Column Definitions

| # | Column Name | Data Type | Permissible Values / Range | Clinical & Operational Description |
|---|---|---|---|---|
| 1 | `token_id` | String | e.g., `GEN-251001-0001` | Unique anonymized token issued at registration counter / kiosk. Format: `{DEPT}-{YYMMDD}-{SEQ}`. |
| 2 | `visit_date` | Date (ISO 8601) | `2025-10-01` to `2025-12-15` | Date of OPD presentation (Monday to Saturday; Sunday closed for routine OPD). |
| 3 | `day_of_week` | Categorical | Monday – Saturday | Day of week. Mondays observe systemic "Monday Surge" (1.35x-1.45x load). Saturdays are half-days. |
| 4 | `arrival_time` | Time (HH:MM:SS) | `07:15:00` to `11:45:00` | Exact physical counter check-in timestamp. |
| 5 | `arrival_minute_of_day` | Integer | 435 to 705 | Arrival time expressed in minutes from midnight (07:15 = 435, 11:45 = 705). Peak rush: 480–615 min. |
| 6 | `department` | Categorical (32) | 32 Clinical Departments | AIIMS Raipur OPD department (e.g., General Medicine, Pediatrics, Cardiology, Orthopedics, Oncology). |
| 7 | `triage_tier` | Integer (1–4) | 1 (Critical) to 4 (Routine) | Clinical Urgency Stratification: <br>• **Tier 1 (Red)**: Resuscitative/Immediate (<2%)<br>• **Tier 2 (Yellow)**: Urgent/Severe pain/Elderly (<18%)<br>• **Tier 3 (Green)**: Semi-urgent symptomatic (~55%)<br>• **Tier 4 (White)**: Routine review/Refill (~25%) |
| 8 | `patient_age` | Integer | 1 to 92 | Age in completed years. |
| 9 | `patient_gender` | Categorical | `M`, `F`, `Other` | Self-reported gender identity. |
| 10 | `is_senior` | Binary (0/1) | 0 or 1 | Indicator if patient $\ge 60$ years (eligible for senior citizen green corridor). |
| 11 | `is_pediatric` | Binary (0/1) | 0 or 1 | Indicator if patient $\le 12$ years. |
| 12 | `patient_origin` | Categorical (7) | Raipur_Urban, Durg_Bhilai, Bilaspur, Bastar, etc. | Geographic origin indicating transit journey (early morning bus arrivals from tribal/rural districts). |
| 13 | `is_referral` | Binary (0/1) | 0 or 1 | Formal referral from District Hospital / Pt. JNM Medical College / Primary Health Centers. |
| 14 | `visit_type` | Categorical (3) | `New_Consultation`, `Follow_Up`, `Report_Review` | Consultation type. Follow-ups and report reviews require shorter doctor contact time. |
| 15 | `active_counters` | Integer | 1 to 8 | Number of attending faculty / Senior Resident consultation chambers active simultaneously in that OPD. |
| 16 | `queue_depth_at_arrival`| Integer | 0 to 65 | Total patients ahead in queue waiting in that department at the moment of token generation. |
| 17 | `temp_celsius` | Float | 18.0 to 36.5 °C | Ambient daytime temperature in Raipur. |
| 18 | `rainfall_mm` | Float | 0.0 to 18.2 mm | Precipitation in mm. |
| 19 | `is_monday` | Binary (0/1) | 0 or 1 | Flag for post-weekend backlog peak. |
| 20 | `is_saturday` | Binary (0/1) | 0 or 1 | Half-day operational flag. |
| 21 | `is_post_holiday` | Binary (0/1) | 0 or 1 | Surge indicator for the day immediately following a public holiday (Dussehra, Diwali, Rajyotsava). |
| 22 | `is_harvest_season` | Binary (0/1) | 0 or 1 | Agricultural cycle signal (November-December Kharif paddy harvest in rural Chhattisgarh). |
| 23 | `has_abha_id` | Binary (0/1) | 0 or 1 | ABDM compliance flag: Ayushman Bharat Health Account linkage status. |
| 24 | `consultation_duration_mins` | Float | 3.0 to 45.0 | Log-normally distributed actual doctor consultation time. |
| 25 | `actual_wait_time_mins` | Float | 1.0 to 180.0 | Ground truth waiting time elapsed between token issuance and entering doctor chamber. |

---

## 3. Mathematical Grounding & Queue Physics
The physical wait time $W_q$ follows multi-server queuing theory with clinical priority stratification:

$$W_q \approx \left(\frac{Q(t)}{s(t) \cdot \mu_{\text{dept}}}\right) \cdot \gamma(\text{triage}) \cdot \rho(\text{rush}) + \epsilon$$

Where:
- $Q(t)$: Queue depth at arrival time $t$.
- $s(t)$: Active consultation counters in the department.
- $\mu_{\text{dept}}$: Departmental service rate (patients/min).
- $\gamma(\text{triage})$: Clinical prioritization weighting factor ($\gamma_1 = 0.08, \gamma_2 = 0.45, \gamma_3 = 1.0, \gamma_4 = 1.25$).
- $\rho(\text{rush})$: Bimodal arrival surge kernel centered around rural bus arrival peaks.
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$: Physical stochastic variance.
