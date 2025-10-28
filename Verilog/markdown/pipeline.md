// Pipeline 4 Array 
| Cycle | array1 input | array2 acc1+input   | array3 acc1+acc2+input         | array4 acc1+acc2+acc3+input          | 输出         |
| :---- | :----------- | :------------------ | :----------------------------- | :----------------------------------- | :--------- |
| 1     | col0–3       | -                   | -                              | -                                    | -          |
| 2     | col16–19     | col0–3 + col4–7     | -                              | -                                    | -          |
| 3     | col32–35     | col16–19 + col20–23 | col0–3 + col4–7 + col8–11      | -                                    | -          |
| 4     | col48–51     | col32–35 + col36–39 | col16–19 + col20–23 + col24–27 | col0–3 + col4–7 + col8–11 + col12–15 | ✅ 输出 tile0 |
| 5     | col64–67     | col48–51 + col52–55 | col32–35 + col36–39 + col40–43 | col16–19 + col20–23 + col24–27 + col28–31 | ✅ 输出 tile1 |
| 6     | col80–83     | col64–67 + col68–71 | col48–51 + col52–55 + col56–59 | col32–35 + col36–39 + col40–43 + col44–47 | ✅ 输出 tile2 |
| 7     | col96–99     | col80–83 + col84–87 | col64–67 + col68–71 + col72–75 | col48–51 + col52–55 + col56–59 + col60–63 | ✅ 输出 tile3 |
//input
| Cycle | array1 输入列 | array2 输入列 | array3 输入列 | array4 输入列 |
| ----- | ---------- | ---------- | ---------- | ---------- |
| 1     | col0–3     | -          | -          | -          |
| 2     | col16–19   | col4–7     | -          | -          |
| 3     | col32–35   | col20–23   | col8–11    | -          |
| 4     | col48–51   | col36–39   | col24–27   | col12–15   |
| 5     | col64–67   | col52–55   | col40–43   | col28–31   |
| …     | …          | …          | …          | …          |

//waveform should be like this
| Cycle | A0_mat               | A1_mat   | A2_mat   | A3_mat   |
| ----- | -------------------- | -------- | -------- | -------- |
| 1     | col0–3               | 0        | 0        | 0        |
| 2     | col16–19             | col4–7   | 0        | 0        |
| 3     | col32–35             | col20–23 | col8–11  | 0        |
| 4     | col48–51             | col36–39 | col24–27 | col12–15 |
| 5     | col64–67             | col52–55 | col40–43 | col28–31 |
| ✅ 输出  | 从第4拍开始每拍输出一个 tile 结果 |          |          |          |
