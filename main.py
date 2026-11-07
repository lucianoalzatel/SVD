# SVG: Compresión de imágenes
# Álgebra Lineal II
# 
# Autores:  Luciano Alzate
#           Santiago González
# 
# Fecha: 2026-11-02

# ======================== IMPORTS ======================== #

import sys                      # (1) 
import numpy as np              # (2) 
import matplotlib.pyplot as plt # (3) 
from PIL import Image           # (4) 
import pandas as pd             # (5) 

# (1) Para recibir entrada por línea de comandos
# (2) Para la manipulación de matrices
# (3) Para elaboración de gráficos
# (4) Para el manejo de imágenes
# (5) Para elaborar tabla de resultados

# ======================== CLASSES ======================== #

class SVD:
    def __init__(self, image: np.ndarray):
        self.A = np.array(image, dtype=np.float64) / 255.0
        self.m, self.n = self.A.shape
        self.U, self.S, self.Vt = np.linalg.svd(self.A, full_matrices=False)
        self.original_memory = self.m * self.n

    def k_approximation(self, k: int) ->np.ndarray:
        if k > len(self.S):
            raise ValueError(f'k={k} excede rango máximo {len(self.S)}')
        return self.U[:, :k] @ np.diag(self.S[:k]) @ self.Vt[:k, :]

    def metrics(self, k: int) -> dict:
        A_k = self.k_approximation(k)

        memory = k * (self.m + self.n + 1)
        storage = (memory / self.original_memory) * 100
        saved = 100.0 - storage

        error_norm2 = np.linalg.norm(self.A - A_k, 2)
        s_k_plus1 = self.S[k] if k < len(self.S) else 0.0

        return {'k' : k,
                'Error Real' : error_norm2,
                'Teórico s_{k+1}' : s_k_plus1,
                'Coincidencia' : np.isclose(error_norm2, s_k_plus1),
                'Flotantes truncados' : memory,
                'Almacenamiento utilizado' : storage,
                'Ahorro espacio' : saved
            }

    def create_report(self, list_k: list[int]) -> pd.DataFrame:
        data = [self.metrics(k) for k in list_k]
        return pd.DataFrame(data)

    def export_figures(self, list_k: list[int], name: str = 'panel.png'):
        num_cols = len(list_k) + 1
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

        axes[0].imshow(self.A, cmap='gray')
        axes[0].set_title(f'Original A\n({self.m}x{self.n})')
        axes[0].axis('off')

        for i, k in enumerate(list_k, start=1):
            A_k = self.k_approximation(k)
            metrics = self.metrics(k)
            axes[i].imshow(np.clip(A_k, 0, 1), cmap='gray')
            axes[i].set_title(f"$A_{{k}}$ (k={k})\n{metrics['Almacenamiento utilizado']:.1f}% de memoria")
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(name, dpi=300, bbox_inches='tight')
        plt.close()

    def export_graph(self, list_k: list[int], name: str = 'valores_singulares.pdf'):
        plt.figure(figsize=(7, 4.5))
        plt.semilogy(range(1, len(self.S) + 1), self.S, 'b-', linewidth=1.5, label=r'Valores singulares $s_j$')

        for k in list_k:
            plt.axvline(x=k, color='r', linestyle='--', alpha=0.5)
            plt.plot(k, self.S[k-1], 'ro')
            plt.text(k, self.S[k-1]*1.2, f'k={k}', fontsize=9, ha='center')

        plt.xlabel('Índice $j$', fontsize=11)
        plt.ylabel('Valor singular $s_j$', fontsize=11)
        # plt.title('Decaimiento de valores singulares y puntos de truncamiento', fontsize=12)
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(name, dpi=300, bbox_inches='tight')
        plt.close()

# ======================== FUNCTIONS ======================== #

def load_grayscale_image(image_path: str) -> np.ndarray:
    image = np.array(Image.open(image_path).convert('L'))
    plt.imsave('original_escala_de_grises.png', image, cmap='gray')

    return image

def calculate_k_values(image: np.ndarray) -> list[int]:
    n = len(image)
    k = (n*n) / (n+n+1)
    
    values = [0.01, 0.10, 0.25, 0.50]
    
    for i, val in enumerate(values):
        values[i] = int(k*val)

    print('===========================')
    print('Matrix size:', n)
    print('Max k:', k)
    print('k values:', values)

    return values

# ======================== MAIN ======================== #

if __name__=='__main__':
    if len(sys.argv) > 1:
        image = load_grayscale_image(sys.argv[1])
    else:
        image = load_grayscale_image('_paloma.jpg')

    svd = SVD(image)

    k_values = calculate_k_values(image)

    results = svd.create_report(k_values)
    print('============ Tabla de resultados ============')
    print(results.to_string(index=False))

    svd.export_figures(k_values)
    svd.export_graph(k_values)
