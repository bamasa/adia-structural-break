# Portfolio card

## English

**Structural-break detection in real time — top 10% of 1,716 (ADIA Lab / CrunchDAO, 2026)**

A streaming detector for structural breaks in univariate time series, scored
by a per-step cross-sectional AUC on the platform's hidden data. Final
0.6299 against a leader at 0.680. The solution is an ensemble of LightGBM
rankers and classifiers over hand-built streaming statistics, dilated causal
networks over their trajectories, and independent members on their own
inputs — the decisive one being the series whitened by its own history
(AR(p), conditional scale, innovation ECDF) with CUSUM, GLR and
Shiryaev-Roberts statistics read on the whitened stream.

The repository documents the whole search, not only the result: 158
experiments with verdicts and reasons, two research surveys run by LLM
agents (what the leaders build; how the data generator makes its breaks),
a shipping protocol with streaming-versus-batch verification, and a method
document on running research with an LLM assistant as the team. Python,
numpy, LightGBM, PyTorch.

## Русский

**Детекция структурных сломов в реальном времени — верхние 10% из 1716 участников (ADIA Lab / CrunchDAO, 2026)**

Потоковый детектор структурных сломов в одномерных временных рядах;
метрика — кросс-секционный AUC по шагам на скрытых данных платформы.
Итог 0.6299 при лидере 0.680. Решение — ансамбль ранкеров и
классификаторов LightGBM на потоковых статистиках, дилатационных
каузальных сетей на их траекториях и независимых членов на собственных
входах; решающий — ряд, отбелённый по собственной истории (AR(p),
условный масштаб, эмпирическое распределение инноваций), на котором
читаются CUSUM, GLR и статистики Ширяева–Робертса.

Репозиторий документирует весь поиск, а не только результат: 158
экспериментов с вердиктами и причинами, два исследовательских обзора,
выполненных LLM-агентами (что строят лидеры; как генератор данных делает
сломы), протокол отправки со сверкой потоковой и пакетной реализаций и
описание метода исследования с LLM-ассистентом в роли команды. Python,
numpy, LightGBM, PyTorch.
