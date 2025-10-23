import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default="./eval_out_log5")
    args = ap.parse_args()
    target_folders = ["Grid174", "Grid182", "Grid201"]
    root_dir = args.root_dir
    ext = ".csv" 

    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(ext):
                csv_path = os.path.join(folder, file)
                png_path = os.path.join(folder, os.path.splitext(file)[0] + ".png")

                try:
                    # ===== read CSV =====
                    df = pd.read_csv(csv_path, header=None)
                    # Handle case where first row is header
                    if not str(df.iloc[0, 0]).replace('.', '', 1).replace('-', '', 1).isdigit():
                       df = pd.read_csv(csv_path, skiprows=1, header=None)
                    
                    if df.shape[1] < 5:
                        print(f"[skip] {csv_path} too few columns.")
                        continue
                    df.columns = ['y_true_x', 'y_true_y', 'y_pred_x', 'y_pred_y', 'err_m']

                    # ===== plot =====
                    plt.figure(figsize=(8, 6))
                    plt.title(f"True vs Predicted Positions\n{file}", fontsize=13)

                    #  Ground truth trajectory (line)
                    plt.plot(df['y_true_x'], df['y_true_y'], 'b-', label='True Trajectory', linewidth=2)

                    #  Predicted positions (scatter)
                    # sc = plt.scatter(df['y_pred_x'], df['y_pred_y'], c=df['err_m'],
                    #                 cmap='plasma', s=2, alpha=0.8, label='Predicted Points')
                    sc = plt.scatter(df['y_pred_x'], df['y_pred_y'], c=range(len(df)),
                                    cmap='plasma', s=2, alpha=0.8, label='Predicted Points')

                    # Colorbar for error
                    cbar = plt.colorbar(sc)
                    cbar.set_label('Error (m)')

                    plt.xlabel("X coordinate")
                    plt.ylabel("Y coordinate")
                    plt.legend()
                    plt.axis('equal')
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()

                    # 保存图片
                    plt.savefig(png_path, dpi=300)
                    plt.close()
                    print(f"[finished] {png_path}")
                except Exception as e:
                    print(f"[error] Failed to process {csv_path} : {e}")


if __name__ == "__main__":
    main()



