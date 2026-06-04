# =============================================================================
# ANÁLISE DE CONURBAÇÃO - LITORAL DE SANTA CATARINA
# Disciplina: Processamento de Imagem
# Classificador: Random Forest (scikit-learn)
# Visualização de rodovias: OSMnx + GeoPandas (suporte)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
from skimage.color import rgb2gray

# scikit-learn — classificador principal
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split

# suporte geográfico (apenas para rodovias)
import osmnx as ox
import geopandas as gpd


# =============================================================================
# 1. CONFIGURAÇÕES GERAIS
# =============================================================================

REGIAO = "Florianópolis, Santa Catarina, Brazil"  # ajuste para sua região
IMAGEM_2015 = "assets/Blumenau2015.png"   # caminho da imagem
IMAGEM_2025 = "assets/Blumenau2025.png"

# Bounding box geográfico da imagem (lon_min, lon_max, lat_min, lat_max)
# Ajuste esses valores conforme os metadados das suas imagens de satélite
BBOX = (-48.70, -48.40, -27.75, -27.45)  # exemplo Florianópolis
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = BBOX


# =============================================================================
# 2. FUNÇÕES AUXILIARES
# =============================================================================

def carregar_imagem(caminho: str) -> np.ndarray:
    """Carrega imagem, converte para escala de cinza e normaliza para uint8."""
    img = io.imread(caminho)
    if img.ndim == 3:
        img = img[:, :, :3]          # remove canal alpha se existir
        img = rgb2gray(img)
    return (img * 255).astype(np.uint8)


def segmentar_camadas(img: np.ndarray) -> dict:
    """
    Divide a imagem em 3 camadas por intensidade de luz:
      - Camada 1 (escura):  0–49   → área não urbanizada / rural
      - Camada 2 (média):  50–149  → urbanização em expansão / periurbano
      - Camada 3 (clara):  150–255 → núcleo urbano consolidado
    """
    return {
        "escura":  (img >= 0)   & (img < 50),
        "media":   (img >= 50)  & (img < 150),
        "clara":   (img >= 150) & (img <= 255),
    }


def extrair_features(img: np.ndarray, tamanho_janela: int = 3) -> np.ndarray:
    """
    Extrai features por pixel usando uma janela local:
      - Valor do pixel central
      - Média da vizinhança
      - Desvio padrão da vizinhança
    Retorna array (n_pixels, 3).
    """
    from scipy.ndimage import uniform_filter, generic_filter

    media   = uniform_filter(img.astype(float), size=tamanho_janela)
    desvio  = generic_filter(img.astype(float), np.std, size=tamanho_janela)

    features = np.stack([
        img.ravel().astype(float),
        media.ravel(),
        desvio.ravel(),
    ], axis=1)
    return features


def gerar_rotulos(img_2015: np.ndarray, img_2025: np.ndarray) -> np.ndarray:
    """
    Gera rótulos automáticos comparando os dois anos:
      0 → diminuiu  (ficou mais escuro — perda de atividade)
      1 → estável   (sem mudança significativa)
      2 → cresceu   (ficou mais claro — expansão urbana)
    Limiar de 20 pontos de intensidade para considerar mudança.
    """
    diff = img_2025.astype(int) - img_2015.astype(int)
    rotulos = np.ones(diff.shape, dtype=int)   # padrão: estável
    rotulos[diff >  20] = 2                    # cresceu
    rotulos[diff < -20] = 0                    # diminuiu
    return rotulos


# =============================================================================
# 3. CARREGAMENTO E PRÉ-PROCESSAMENTO
# =============================================================================

print("=" * 60)
print("CARREGANDO IMAGENS...")
print("=" * 60)

img_2015 = carregar_imagem(IMAGEM_2015)
img_2025 = carregar_imagem(IMAGEM_2025)

camadas_2015 = segmentar_camadas(img_2015)
camadas_2025 = segmentar_camadas(img_2025)

print(f"Resolução das imagens: {img_2015.shape}")


# =============================================================================
# 4. VISUALIZAÇÃO DAS CAMADAS (2015 e 2025)
# =============================================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Segmentação por Camadas de Luminosidade", fontsize=14, fontweight='bold')

nomes = ["Original", "Escura (rural)", "Média (periurbano)", "Clara (urbano)"]
dados_2015 = [img_2015, camadas_2015["escura"], camadas_2015["media"], camadas_2015["clara"]]
dados_2025 = [img_2025, camadas_2025["escura"], camadas_2025["media"], camadas_2025["clara"]]

for col in range(4):
    axes[0, col].imshow(dados_2015[col], cmap='gray')
    axes[0, col].set_title(f"2015 — {nomes[col]}", fontsize=9)
    axes[0, col].axis('off')

    axes[1, col].imshow(dados_2025[col], cmap='gray')
    axes[1, col].set_title(f"2025 — {nomes[col]}", fontsize=9)
    axes[1, col].axis('off')

plt.tight_layout()
plt.savefig("fig1_camadas.png", dpi=150, bbox_inches='tight')
plt.show()
print("✔ Figura 1 salva: fig1_camadas.png")


# =============================================================================
# 5. MAPA DE DIFERENÇA TEMPORAL
# =============================================================================

min_h = min(img_2025.shape[0], img_2015.shape[0])
min_w = min(img_2025.shape[1], img_2015.shape[1])

# 2. Recorte AMBAS para o tamanho mínimo comum
# Isso garante que elas tenham exatamente o mesmo formato (min_h, min_w)
img_2025_ajustada = img_2025[:min_h, :min_w]
img_2015_ajustada = img_2015[:min_h, :min_w]

# 3. Agora a subtração funcionará
diff = img_2025_ajustada.astype(int) - img_2015_ajustada.astype(int)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(diff, cmap='RdYlGn', vmin=-100, vmax=100)
plt.colorbar(im, ax=ax, label='Variação de Intensidade (2015 → 2025)')
ax.set_title("Mapa de Crescimento Urbano\n(verde = cresceu | vermelho = diminuiu)", fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig("fig2_diferenca.png", dpi=150, bbox_inches='tight')
plt.show()
print("✔ Figura 2 salva: fig2_diferenca.png")


# =============================================================================
# 6. CLASSIFICADOR RANDOM FOREST (scikit-learn)
# =============================================================================

print("\n" + "=" * 60)
print("TREINANDO RANDOM FOREST...")
print("=" * 60)

# --- features e rótulos ---
X = extrair_features(img_2015)           # features baseadas em 2015
y = gerar_rotulos(img_2015, img_2025)    # rótulos: o que mudou até 2025
y_flat = y.ravel()

# --- divisão treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_flat, test_size=0.3, random_state=42, stratify=y_flat
)

print(f"Amostras treino: {len(X_train):,} | teste: {len(X_test):,}")
print(f"Classes — 0:diminuiu  1:estável  2:cresceu")
print(f"Distribuição treino: {np.bincount(y_train)}")

# --- treino ---
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1       # usa todos os núcleos disponíveis
)
rf.fit(X_train, y_train)

# --- avaliação ---
y_pred = rf.predict(X_test)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred,
      target_names=["Diminuiu", "Estável", "Cresceu"]))

# --- matriz de confusão ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Diminuiu", "Estável", "Cresceu"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Matriz de Confusão — Random Forest", fontweight='bold')
plt.tight_layout()
plt.savefig("fig3_matriz_confusao.png", dpi=150, bbox_inches='tight')
plt.show()
print("✔ Figura 3 salva: fig3_matriz_confusao.png")

# --- mapa de predição ---
X_full = extrair_features(img_2015)
mapa_pred = rf.predict(X_full).reshape(img_2015.shape)

cores = np.zeros((*img_2015.shape, 3), dtype=np.uint8)
cores[mapa_pred == 0] = [220,  50,  50]   # vermelho  — diminuiu
cores[mapa_pred == 1] = [ 50,  50,  50]   # cinza     — estável
cores[mapa_pred == 2] = [ 50, 200,  80]   # verde     — cresceu

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cores)
ax.set_title("Classificação Random Forest — Deslocamento do Crescimento Urbano",
             fontweight='bold')
patches = [
    mpatches.Patch(color='#DC3232', label='Diminuiu'),
    mpatches.Patch(color='#323232', label='Estável'),
    mpatches.Patch(color='#32C850', label='Cresceu'),
]
ax.legend(handles=patches, loc='lower right', fontsize=9)
ax.axis('off')
plt.tight_layout()
plt.savefig("fig4_mapa_classificacao.png", dpi=150, bbox_inches='tight')
plt.show()
print("✔ Figura 4 salva: fig4_mapa_classificacao.png")


# =============================================================================
# 7. ANÁLISE DBSCAN — DISTÂNCIA ENTRE NÚCLEOS URBANOS
# =============================================================================

print("\n" + "=" * 60)
print("ANÁLISE DBSCAN — APROXIMAÇÃO ENTRE CIDADES...")
print("=" * 60)

def analisar_dbscan(img: np.ndarray, ano: str):
    """Aplica DBSCAN sobre pixels luminosos e retorna clusters e distância mínima."""
    coords = np.column_stack(np.where(img > 50))   # apenas pixels com luz
    if len(coords) == 0:
        print(f"{ano}: nenhum pixel luminoso encontrado.")
        return None, None

    db = DBSCAN(eps=5, min_samples=20, n_jobs=-1)
    labels = db.fit_predict(coords)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"{ano}: {n_clusters} clusters encontrados ({np.sum(labels == -1):,} ruído)")

    # distância mínima entre os dois maiores clusters
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) < 2:
        print(f"{ano}: menos de 2 clusters — impossível calcular distância.")
        return coords, labels

    top2 = unique[np.argsort(counts)[-2:]]
    A = coords[labels == top2[0]]
    B = coords[labels == top2[1]]

    dist_min = euclidean_distances(A, B).min()
    print(f"{ano}: distância mínima entre os 2 maiores clusters = {dist_min:.2f} px")
    return coords, labels

coords_2015, labels_2015 = analisar_dbscan(img_2015, "2015")
coords_2025, labels_2025 = analisar_dbscan(img_2025, "2025")


# =============================================================================
# 8. ANÁLISE DE CRESCIMENTO POR CAMADA
# =============================================================================

print("\n" + "=" * 60)
print("CRESCIMENTO POR CAMADA DE LUMINOSIDADE")
print("=" * 60)

resultados = {}
for camada in ["escura", "media", "clara"]:
    a2015 = int(np.sum(camadas_2015[camada]))
    a2025 = int(np.sum(camadas_2025[camada]))
    cresc  = ((a2025 - a2015) / a2015) * 100 if a2015 > 0 else 0
    resultados[camada] = {"2015": a2015, "2025": a2025, "crescimento": cresc}
    print(f"  {camada:8s}: 2015={a2015:8,}px | 2025={a2025:8,}px | Δ={cresc:+.1f}%")

# gráfico de barras
fig, ax = plt.subplots(figsize=(8, 5))
camadas_nomes = ["Escura\n(rural)", "Média\n(periurbano)", "Clara\n(urbano)"]
crescimentos  = [resultados[c]["crescimento"] for c in ["escura", "media", "clara"]]
cores_barras  = ['#5599DD', '#FFAA33', '#EE4444']

bars = ax.bar(camadas_nomes, crescimentos, color=cores_barras, edgecolor='black', width=0.5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel("Crescimento (%)", fontsize=11)
ax.set_title("Variação de Área por Camada de Luminosidade (2015 → 2025)",
             fontsize=12, fontweight='bold')

for bar, val in zip(bars, crescimentos):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:+.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("fig5_crescimento_camadas.png", dpi=150, bbox_inches='tight')
plt.show()
print("✔ Figura 5 salva: fig5_crescimento_camadas.png")


# =============================================================================
# 9. PERFIL DE LUMINOSIDADE (TRANSECTO ENTRE CIDADES)
# =============================================================================

from skimage.measure import profile_line

# Ajuste os pontos (linha, coluna) conforme a posição das cidades na imagem
# Exemplo: ponto A = norte da imagem, ponto B = sul
ponto_A = (int(img_2015.shape[0] * 0.15), int(img_2015.shape[1] * 0.5))
ponto_B = (int(img_2015.shape[0] * 0.85), int(img_2015.shape[1] * 0.5))

perfil_2015 = profile_line(img_2015, ponto_A, ponto_B, linewidth=5)
perfil_2025 = profile_line(img_2025, ponto_A, ponto_B, linewidth=5)

fig, ax = plt.subplots(figsize=(10, 4))
x = np.linspace(0, 100, len(perfil_2015))
ax.fill_between(x, perfil_2015, alpha=0.3, color='blue')
ax.fill_between(x, perfil_2025, alpha=0.3, color='orange')
ax.plot(x, perfil_2015, color='blue',   linewidth=2, label='2015')
ax.plot(x, perfil_2025, color='orange', linewidth=2, label='2025')
ax.set_xlabel("Posição ao longo do transecto (%)")
ax.set_ylabel("Intensidade de Luz (0–255)")
ax.set_title("Perfil de Luminosidade — Transecto Norte–Sul\n"
             "(elevação do vale entre cidades indica conurbação)", fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig6_transecto.png", dpi=150, bbox_inches='tight')
plt.show()
print("✔ Figura 6 salva: fig6_transecto.png")


# =============================================================================
# 10. SOBREPOSIÇÃO DE RODOVIAS (OSMnx — suporte visual)
# =============================================================================

print("\n" + "=" * 60)
print("BAIXANDO MALHA VIÁRIA (OSMnx)...")
print("=" * 60)

try:
    # baixa apenas rodovias principais (motorway, trunk, primary, secondary)
    cf = '["highway"~"motorway|trunk|primary|secondary"]'
    G = ox.graph_from_place(REGIAO, custom_filter=cf, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
    print(f"✔ {len(edges)} segmentos de rodovia carregados.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Luzes Noturnas + Rodovias Principais", fontsize=14, fontweight='bold')

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    for ax, img_ano, ano in [(axes[0], img_2015, "2015"), (axes[1], img_2025, "2025")]:
        ax.imshow(img_ano, cmap='gray', extent=extent, origin='upper',
                  aspect='auto')
        edges.plot(ax=ax, color='red', linewidth=0.8, alpha=0.85)
        ax.set_title(f"{ano}", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig("fig7_rodovias.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✔ Figura 7 salva: fig7_rodovias.png")

except Exception as e:
    print(f"⚠ Não foi possível baixar rodovias: {e}")
    print("  Verifique a conexão com internet e o nome da REGIAO.")


# =============================================================================
# 11. IMPORTÂNCIA DAS FEATURES (RANDOM FOREST)
# =============================================================================

importancias = rf.feature_importances_
nomes_feat   = ["Intensidade pixel", "Média vizinhança", "Desvio padrão viz."]

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(nomes_feat, importancias, color=['#4477AA', '#66AADD', '#AACCEE'],
        edgecolor='black')
ax.set_xlabel("Importância")
ax.set_title("Importância das Features — Random Forest", fontweight='bold')
for i, v in enumerate(importancias):
    ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=10)
plt.tight_layout()
plt.savefig("fig8_importancia_features.png", dpi=150, bbox_inches='tight')
plt.show()
print("✔ Figura 8 salva: fig8_importancia_features.png")


# =============================================================================
# 12. RESUMO FINAL
# =============================================================================

print("\n" + "=" * 60)
print("RESUMO DA ANÁLISE")
print("=" * 60)
print(f"  Acurácia Random Forest: {rf.score(X_test, y_test):.4f}")
for camada, r in resultados.items():
    print(f"  Camada {camada:8s}: Δ={r['crescimento']:+.1f}%")
print("=" * 60)
print("Figuras geradas:")
for i, nome in enumerate([
    "Camadas de luminosidade",
    "Mapa de diferença temporal",
    "Matriz de confusão RF",
    "Mapa de classificação RF",
    "Crescimento por camada",
    "Perfil de transecto",
    "Sobreposição de rodovias",
    "Importância das features",
], start=1):
    print(f"  fig{i}_{nome.lower().replace(' ', '_')}.png")