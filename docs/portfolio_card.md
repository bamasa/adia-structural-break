# Portfolio card

## English

**Real-time structural-break detection — about rank 160 of 1,716 registered participants on the public leaderboard, final evaluation pending (ADIA Lab / CrunchDAO Structural Break Challenge, Real-Time Edition, 2026). Solo entry, August–October 2026. Python, NumPy, LightGBM, PyTorch.**

A streaming detector that, after every new observation of a univariate time series, scores whether a permanent break has already occurred; the metric is a per-step cross-sectional AUC on the platform's hidden data. Closed at 0.6299 against a leader at 0.680. The solution is an ensemble of LightGBM rankers and classifiers over hand-built streaming statistics, dilated causal networks over their trajectories, and independent members on their own inputs; the decisive one whitens the series by its own history (AR(p), conditional scale, innovation ECDF) and reads CUSUM, GLR and Shiryaev–Roberts statistics on the whitened stream. The repository records the whole search, not only the result: 158 experiments with verdicts and reasons, two research surveys run by LLM agents (what the leaders build; how the data generator makes its breaks), a shipping protocol with streaming-versus-batch verification, and a method note on directing an LLM assistant as the research team.

## Русский

**Детекция структурных сломов в реальном времени — около 160-го места из 1716 зарегистрированных участников на публичной доске, финальная оценка впереди (ADIA Lab / CrunchDAO, Structural Break Challenge, Real-Time Edition, 2026). Индивидуальное участие, август–октябрь 2026. Python, NumPy, LightGBM, PyTorch.**

Потоковый детектор, который после каждого нового наблюдения одномерного временного ряда оценивает, произошёл ли уже необратимый слом; метрика — кросс-секционный AUC по шагам на скрытых данных платформы. Итог 0.6299 при лидере 0.680. Решение — ансамбль ранкеров и классификаторов LightGBM на потоковых статистиках, дилатационных каузальных сетей на их траекториях и независимых членов на собственных входах; решающий член отбеливает ряд по его собственной истории (AR(p), условный масштаб, эмпирическое распределение инноваций) и читает на отбелённом потоке CUSUM, GLR и статистики Ширяева–Робертса. Репозиторий фиксирует весь поиск, а не только результат: 158 экспериментов с вердиктами и причинами, два исследовательских обзора, выполненных LLM-агентами (что строят лидеры; как генератор данных делает сломы), протокол отправки со сверкой потоковой и пакетной реализаций и описание метода работы с LLM-ассистентом в роли исследовательской команды.
