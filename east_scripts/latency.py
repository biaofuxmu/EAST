import re
import torch


def count_length(text, lang):
    if lang == "Chinese":
        return len(text)
    else:
        return len(text.split())


def compute_delays(read_content, translate_content, src_lang, tgt_lang):
    # 去除每个匹配内容两端的空格
    read_content = [content.strip() for content in read_content]
    translate_content = [content.strip() for content in translate_content]

    # 计算每个 read_content 中内容的累积长度
    cumulative_lengths = []
    cumulative_length = 0
    for content in read_content:
        cumulative_length += count_length(content, src_lang)
        cumulative_lengths.append(cumulative_length)

    # 获取 cumulative_lengths 对应 translate_content 的长度
    delays = []
    for i in range(len(translate_content)):
        if tgt_lang == "Chinese":
            translate_tokens = list(translate_content[i])
        else:
            translate_tokens = translate_content[i].split()
        for j in range(len(translate_tokens)):
            delays.append(cumulative_lengths[i])

    return delays, delays[-1]

# copy from https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py#L115
def AverageLagging(delays, source_length, target_length):
    """
    Average Lagging (AL) from
    `STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework <https://arxiv.org/abs/1810.08398>`_

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        AL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{|Y|}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length
    """

    if delays[0] > source_length:
        return delays[0]

    AL = 0
    gamma = target_length / source_length
    tau = 0
    for t_minus_1, d in enumerate(delays):
        AL += d - t_minus_1 / gamma
        tau = t_minus_1 + 1

        if d >= source_length:
            break
    AL /= tau
    return AL

# copy from https://github.com/facebookresearch/SimulEval/blob/main/simuleval/evaluator/scorers/latency_scorer.py#L167
def LengthAdaptiveAverageLagging(delays, source_length, target_length):
    r"""
    Length Adaptive Average Lagging (LAAL) as proposed in
    `CUNI-KIT System for Simultaneous Speech Translation Task at IWSLT 2022
    <https://arxiv.org/abs/2204.06028>`_.
    The name was suggested in `Over-Generation Cannot Be Rewarded:
    Length-Adaptive Average Lagging for Simultaneous Speech Translation
    <https://arxiv.org/abs/2206.05807>`_.
    It is the original Average Lagging as proposed in
    `Controllable Latency using Prefix-to-Prefix Framework
    <https://arxiv.org/abs/1810.08398>`_
    but is robust to the length difference between the hypothesis and reference.

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        LAAL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{max(|Y|,|Y*|)}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length, and :math:`|Y*|` is the length of the hypothesis.
    """

    if delays[0] > source_length:
        return delays[0]

    LAAL = 0
    gamma = max(len(delays), target_length) / source_length
    tau = 0
    for t_minus_1, d in enumerate(delays):
        LAAL += d - t_minus_1 / gamma
        tau = t_minus_1 + 1

        if d >= source_length:
            break
    LAAL /= tau
    return LAAL