# Previsão de Risco de Churn em Call Center — Áudio (AppTek)

**Autor:** gonaldomichael@gmail.com · **Data:** 2026-11-15 · **SEED:** 42

## Fonte dos dados
AppTek Callcenter Dialogues (Hugging Face: `apptek-com/apptek_callcenter_dialogues`,
config `diarization`), licença CC BY-SA 4.0. 873 conversas role-played em inglês,
151,1 h, 16 domínios, 14 acentos, 94 679 segmentos com diarização, transcrição,
speaker_id e género. Atribuição obrigatória: AppTek / Hugging Face.

## Problema e label
Prever se o cliente indica RISCO DE CHURN a partir de features extraídas do
áudio e dos metadados — NUNCA do texto do cliente (que define o label;
anti-leakage rigoroso). Label proxy (união): churn explícito, frustração
forte, escalonamento, ameaça de saída → 79/873 conversas (9,0%).

## Método
- 51 features: comportamentais (interrupções, latência, ritmo, turnos),
  acústicas do cliente (MFCC, RMS, pitch, ZCR, centróide — librosa) e do
  texto do AGENTE (pedidos de desculpa, empatia).
- Modelo: Regressão Logística (C=0,03, class_weight balanced).
- Avaliação honesta: split e CV AGRUPADOS POR FALANTE (vozes nunca vistas).

## Métricas (verificadas por execução)
- Hold-out disjunto por falante (42 conversas): F1 0,698 | AUC 0,708 |
  precisão 83,3% | recall 60,0% (meta F1 0,70 falhada por 0,002).
- Out-of-fold 5-fold agrupada (as 160 deste dataset): F1 0.632 |
  AUC 0.651 | precisão 64.5% | recall 62.0%.
- Uso recomendado: ranker de priorização (limiar 0,50; top-50% apanha 68%
  dos casos com 81% de precisão).

## Ficheiros
- `churn_features_oof_160.csv` — 160 conversas com features, `churn_label`,
  `churn_score_oof` (previsão out-of-fold) e `previsao_risco` (limiar 0,50).
- `conversas_873.csv` — as 873 conversas com agregados e `churn_label`.

## Dicionário (grupos de colunas)
- `id_conversa`, `acento`, `dominio`, `genero_*`: metadados.
- `duracao_s`, `n_segmentos`, `dur_fala_*`, `turnos_*`, `interrupcoes_*`,
  `latencia_*`, `pct_silencio`, `ritmo_*`, `n_palavras_*`: comportamentais.
- `ag_*`: padrões do texto do agente (desculpas, empatia, agradecimentos).
- `ac_*`: acústicas (MFCC1–13, RMS, pitch, ZCR, centróide; `ac_cli_*` = só
  nos segmentos do cliente).
- `churn_label`: alvo. `churn_score_oof`/`previsao_risco`: previsões OOF.

## Limitações
Conversas role-played (não produção); label proxy derivado do texto; áudio
processado numa amostra estratificada de 160 conversas; n pequeno →
intervalos largos. Detalhes completos no notebook original.
