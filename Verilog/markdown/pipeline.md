# 🚀 Mamba SSM – 4×4×4 Pipeline Array Memory Scheduling  
_Memory Access Pattern & Bank Mapping Overview_

---

## 🧩 Pipeline Execution Schedule  

| **Cycle** | **Array1 Input** | **Array2 (acc1+input)** | **Array3 (acc1+acc2+input)** | **Array4 (acc1+acc2+acc3+input)** | **Output** |
|:----------|:-----------------|:-------------------------|:------------------------------|:----------------------------------|:------------|
| 1 | col0–3 | - | - | - | - |
| 2 | col16–19 | col0–3 + col4–7 | - | - | - |
| 3 | col32–35 | col16–19 + col20–23 | col0–3 + col4–7 + col8–11 | - | - |
| 4 | col48–51 | col32–35 + col36–39 | col16–19 + col20–23 + col24–27 | col0–3 + col4–7 + col8–11 + col12–15 | ✅ tile0 output |
| 5 | col64–67 | col48–51 + col52–55 | col32–35 + col36–39 + col40–43 | col16–19 + col20–23 + col24–27 + col28–31 | ✅ tile1 output |
| 6 | col80–83 | col64–67 + col68–71 | col48–51 + col52–55 + col56–59 | col32–35 + col36–39 + col40–43 + col44–47 | ✅ tile2 output |
| 7 | col96–99 | col80–83 + col84–87 | col64–67 + col68–71 + col72–75 | col48–51 + col52–55 + col56–59 + col60–63 | ✅ tile3 output |

---

## 🧮 Input Column Scheduling  

| **Cycle** | **Array1 Columns** | **Array2 Columns** | **Array3 Columns** | **Array4 Columns** |
|:----------|:------------------|:------------------|:------------------|:------------------|
| 1 | col0–3 | - | - | - |
| 2 | col16–19 | col4–7 | - | - |
| 3 | col32–35 | col20–23 | col8–11 | - |
| 4 | col48–51 | col36–39 | col24–27 | col12–15 |
| 5 | col64–67 | col52–55 | col40–43 | col28–31 |
| 6 | col80–83 | col68–71 | col56–59 | col44–47 |
| 7 | col96–99 | col84–87 | col72–75 | col60–63 |
| 8 | col112–115 | col100–103 | col88–91 | col76–79 |
| 9 | col128–131 | col116–119 | col104–107 | col92–95 |
| 10 | col144–147 | col132–135 | col120–123 | col108–111 |
| … | … | … | … | … |

> 🔹 Each array fetches one 4×4 column block per cycle.  
> 🔹 Column spacing between adjacent arrays = 12 columns (3 blocks).  
> 🔹 From Cycle 4 onward, one tile result is produced per cycle.


---

## 📈 Column Index Progression  

| **Array** | **Column Start Points** | **Δ (Increment)** |
|:----------|:-----------------------|:------------------|
| Array1 | 0 → 16 → 32 → 48 → 64 | +16 each step |
| Array2 | 4 → 20 → 36 → 52 | +16 each step |
| Array3 | 8 → 24 → 40 | +16 each step |
| Array4 | 12 → 28 | +16 each step |

---

## 🔍 Column Spacing Between Arrays  

| **Cycle** | **Array1→Array2 Δ** | **Array2→Array3 Δ** | **Array3→Array4 Δ** |
|:----------|:--------------------|:--------------------|:--------------------|
| 2 | 16−4 = **12** | - | - |
| 3 | 32−20 = **12** | 20−8 = **12** | - |
| 4 | 48−36 = **12** | 36−24 = **12** | 24−12 = **12** |
| 5 | 64−52 = **12** | 52−40 = **12** | 40−28 = **12** |

> ✅ **Conclusion:**  
> Column spacing between adjacent arrays within the same cycle = **12 columns**.

---

## 🧠 Bank Design Summary  

**Bank Count:**  
$$
N_\text{bank} = n_\text{array} \times \text{block\_offset}
$$

- \( n_\text{array} = 4 \)  
- \( \text{block\_offset} = 3 \)  (since column spacing = 12 = 3 blocks)  
→ ✅ \( N_\text{bank} = 4 × 3 = 12 \)

**Bank Mapping Function:**  
$$
\text{bank\_id} = (\lfloor \tfrac{col}{4} \rfloor + 3 × \text{array\_id}) \bmod N_\text{bank}
$$

---

## 🧱 Bank–Column Mapping  

| **Bank ID** | **col_block_id (4-column block IDs)** | **Column Range** |
|:-------------|:-------------------------------------|:-----------------|
| **bank0** | 0, 12, 24, 36, 48, 60 | col0–3, 48–51, 96–99, 144–147, 192–195, 240–243 |
| **bank1** | 1, 13, 25, 37, 49, 61 | col4–7, 52–55, 100–103, 148–151, 196–199, 244–247 |
| **bank2** | 2, 14, 26, 38, 50, 62 | col8–11, 56–59, 104–107, 152–155, 200–203, 248–251 |
| **bank3** | 3, 15, 27, 39, 51, 63 | col12–15, 60–63, 108–111, 156–159, 204–207, 252–255 |
| **bank4** | 4, 16, 28, 40, 52 | col16–19, 64–67, 112–115, 160–163, 208–211 |
| **bank5** | 5, 17, 29, 41, 53 | col20–23, 68–71, 116–119, 164–167, 212–215 |
| **bank6** | 6, 18, 30, 42, 54 | col24–27, 72–75, 120–123, 168–171, 216–219 |
| **bank7** | 7, 19, 31, 43, 55 | col28–31, 76–79, 124–127, 172–175, 220–223 |
| **bank8** | 8, 20, 32, 44, 56 | col32–35, 80–83, 128–131, 176–179, 224–227 |
| **bank9** | 9, 21, 33, 45, 57 | col36–39, 84–87, 132–135, 180–183, 228–231 |
| **bank10** | 10, 22, 34, 46, 58 | col40–43, 88–91, 136–139, 184–187, 232–235 |
| **bank11** | 11, 23, 35, 47, 59 | col44–47, 92–95, 140–143, 188–191, 236–239 |

> ✅ Each bank stores every 12th 4×4 column block (stride = 12).  
> ✅ Round-robin distribution guarantees conflict-free parallel reads.

---
##🕓 Timeline + Bank Access Visualization
---
| **Cycle** | **Array1 → bank** | **Array2 → bank** | **Array3 → bank** | **Array4 → bank** | **Banks Active (total 4)** |
|:----------|:------------------|:------------------|:------------------|:------------------|:---------------------------|
| 1 | bank0 | - | - | - | bank0 |
| 2 | bank4 | bank1 | - | - | bank4, bank1 |
| 3 | bank8 | bank5 | bank2 | - | bank8, bank5, bank2 |
| 4 | bank0 | bank9 | bank6 | bank3 | bank0, bank9, bank6, bank3 |
| 5 | bank4 | bank10 | bank7 | bank1 | bank4, bank10, bank7, bank1 |
| 6 | bank8 | bank11 | bank4 | bank2 | bank8, bank11, bank4, bank2 |
| 7 | bank0 | bank5 | bank8 | bank3 | bank0, bank5, bank8, bank3 |
| 8 | bank4 | bank9 | bank6 | bank1 | bank4, bank9, bank6, bank1 |
| 9 | bank8 | bank10 | bank7 | bank2 | bank8, bank10, bank7, bank2 |
| 10 | bank0 | bank11 | bank4 | bank3 | bank0, bank11, bank4, bank3 |

> 🧠 **Interpretation:**
> - Each cycle activates **4 of 12 banks** (one per array).  
> - Pattern repeats every 4 cycles with stride = 3 banks per array.  
> - Guarantees **conflict-free**, full-bandwidth parallel read for all 4 arrays.  
> - From Cycle 4 onward, one 4×4 tile result is produced each cycle.

---
## 📊 Expected Wave
form  

| **Cycle** | **Array1 (A0_mat)** | **Array2 (A1_mat)** | **Array3 (A2_mat)** | **Array4 (A3_mat)** |
|:----------|:--------------------|:--------------------|:--------------------|:--------------------|
| 1 | col0–3 | 0 | 0 | 0 |
| 2 | col16–19 | col4–7 | 0 | 0 |
| 3 | col32–35 | col20–23 | col8–11 | 0 |
| 4 | col48–51 | col36–39 | col24–27 | col12–15 |
| 5 | col64–67 | col52–55 | col40–43 | col28–31 |

✅ **Output Behavior:**  
From Cycle 4 onward, one tile result is produced every cycle.

---

### 🧾 Notes
- Each 4×4 block = 16 weights (aligned with MAC array width).  
- 12-bank mapping ensures **conflict-free** parallel access for 4 arrays.  
- Mapping function `(col_blk + 3×array_id) % 12` provides even bank utilization.  
- Proper **bank interleaving** is key to achieving simultaneous row-and-column fetching.
