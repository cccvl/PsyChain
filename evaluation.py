import os
import json
import argparse
import asyncio
from tqdm import tqdm
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient



model_gpt4o = OpenAIChatCompletionClient(
    model="ENTER_MODEL_NAME",
    api_key="ENTER_API_KEY",
    base_url="ENTER_BASE_URL",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        "structured_output": True,
        "multiple_system_messages": True,
    },
)

model_deepseek32 = OpenAIChatCompletionClient(
    model="ENTER_MODEL_NAME",
    api_key="ENTER_API_KEY",
    base_url="ENTER_BASE_URL",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        "structured_output": True,
        "multiple_system_messages": True,
    },
)


prompt_active_listening = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对输入的心理咨询多轮对话语料进行质量评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
评估心理咨询过程中，咨询师发言的积极倾听程度，并按照为你定义的评分标准，进行1-5分的打分。
你要重点关注咨询师的发言，分析咨询师积极倾听程度并打分。

# 积极倾听的含义
咨询师在倾听过程中给出表现出的尊重与积极性。咨询师会仔细聆听来访者的发言，确认其主要关切点和情绪状态。这有助于建立关系和信任，同时也能让来访者感到自己被充分倾听。

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
• 5：咨询师专注倾听，不打断，表现出充分的理解，并准确反映客户的感受和担忧。
• 4：咨询师倾听良好，但偶尔会忽略一些小细节或轻微打断。
• 3：咨询师在倾听，但难以抓住关键细节，或对客户交流中的某些方面存在误解。
• 2：咨询师部分倾听，经常错过重要线索或未能把握主要问题。
• 1：咨询师没有积极倾听，经常打断或对客户的表述表现出很少的投入。

# 输出
直接输出打分的结果，即一个1到5之间的整数，不要给出任何其他输出。

"""

prompt_cognitive_restructuring = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对输入的心理咨询多轮对话语料进行质量评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
评估心理咨询过程中，咨询师通过发言帮助来访者实现认知重构的程度，并按照为你定义的评分标准，进行1-5分的打分。
你要重点关注咨询师的发言，分析咨询师引导认知重构的能力并打分。

# 认知重构的含义
认知重构包括帮助来访者识别并挑战其扭曲或不切实际的思维模式。咨询师通过认知重构协助来访者打破原有消极或不适应的想法，培养更现实和有益的认知模式，从而促进来访者情绪健康。

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
• 5：咨询师巧妙地帮助来访者识别出扭曲的想法，并温和地引导他们形成更为平衡、现实或积极的观念。
• 4：咨询师帮助挑战扭曲的思维，但可能无法始终提供明确的替代方案或深刻的见解。
• 3：咨询师提供了一些认知重构的方法，但整个过程显得不完整，或者对思维模式的探讨不够充分。
• 2：咨询师很少进行认知重构，为挑战负面思维提供的指导也很有限。
• 1：咨询师不处理认知扭曲，或者未能帮助来访者改变无益的思维模式。

# 输出
直接输出打分的结果，即一个1到5之间的整数，不要给出任何其他输出。
"""

prompt_empathy = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对输入的心理咨询多轮对话语料进行质量评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
评估心理咨询过程中，咨询师展现出的共情表达能力，并按照为你定义的评分标准，进行1-5分的打分。
你要重点关注咨询师的发言，分析咨询师共情表达能力并打分。

# 共情的含义
共情（同理心）指的是咨询师具备的一种能力，即能够理解、共鸣并认可来访者的情绪和经历。这不仅包括识别来访者的情感，还包括在必要时传达一种深刻的、充满情感的理解与支持的态度。
注意，共情不代表盲目的谄媚和认可，不能体现为对来访者一切行为的附和与肯定，而更应该侧重于对于其情绪困境的理解和支持。

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
• 5：咨询师展现出极深的同理心，始终能够以一种能促进双方联系的方式，认可并恰当回应客户的感受与经历。
• 4：咨询师具备同理心，但在某些时刻，这种同理心可能缺乏深度或清晰度。
• 3：咨询师表现出基本的同理心，但对情感的理解感觉有些疏远或不完整。
• 2：咨询师难以展现同理心，对情感的理解显得肤浅或不足，或出现过分谄媚的情况。
• 1：咨询师没有表现出同理心，或者对客户的感情经历似乎漠不关心。

# 输出
直接输出打分的结果，即一个1到5之间的整数，不要给出任何其他输出。
"""

prompt_strategy_professionalism = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对输入的心理咨询多轮对话语料进行质量评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
评估心理咨询过程中，咨询师发言策略专业性的水平，并按照为你定义的评分标准，进行1-5分的打分。
你要重点关注咨询师的发言，分析咨询师策略专业性的水平并打分。

# 策略专业性的含义
策略专业性指的是咨询师在对话过程中专业能力的展现，即有效运用专业的心理咨询技术和会话策略（如提问与探索、反馈与总结、问题解决与引导），并保持对话的专业性和引导性，严格遵循专业的咨询框架和具体的实施流程。

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
• 5：咨询师精湛运用专业策略和技术，有效引导来访者进行深度自我探索和自主决策，遵循专业心理咨询框架，展现出清晰的咨询流程，并给出专业的方案或干预措施。
• 4：咨询师能够运用恰当的策略和专业技术，有效引导来访者，对话基本符合专业心理咨询框架，但在具体实施细节上仍有改进空间。
• 3：咨询师运用了部分恰当的策略和技术，能在一定程度上引导来访者，但缺乏连贯性、专业性或对咨询框架的遵循不足。
• 2：咨询师尝试运用部分策略或技术，但未能有效引导来访者，或对话缺乏专业框架。
• 1：咨询师未能运用恰当策略或专业技术，对话混乱，完全偏离咨询目标。

# 输出
直接输出打分的结果，即一个1到5之间的整数，不要给出任何其他输出。
"""

prompt_guiding_question = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对输入的心理咨询多轮对话语料进行质量评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
评估心理咨询过程中，咨询师使用开放性问题与引导性问题的程度，并按照为你定义的评分标准，进行1-5分的打分。
你要重点关注咨询师的发言，分析咨询师使用开放性与引导性问题的技巧并打分。

# 开放性问题与引导性问题的含义
-开放性问题：开放式问题旨在鼓励来访者更开放地探讨自己的想法、感受和经历。这些问题通常以“如何”“什么”或“你能再详细说说吗”这类表述开头，促进来访者给出详尽的回答，成为探索其深层心理问题的重要基础。
-引导性问题：引导式提问指的是咨询师通过提出有针对性的问题，帮助来访者明确其关注点或聚焦于特定目标。出色运用引导性问题有助于帮来访者找到针对具体问题的解决办法，通常会促使他们对自身经历的某些特定方面进行更深入的思考。

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
• 5：咨询师精湛地融合运用开放性问题和引导性问题，提问富有策略性和引导性，持续激发来访者深度自我探索，并高效引导其明确问题、形成洞察并制定可行的解决方案。
• 4：咨询师能有效运用开放性问题和引导性问题，并能根据对话需要进行调整，成功鼓励来访者进行探索，并能适时引导其聚焦问题或寻找解决方案，不过尚未形成灵活的融合运用能力。
• 3：咨询师能够运用开放性问题和引导性问题，但在运用时存在不一致性或深度不足，有时能促进来访者思考，但未能充分发挥其引导和探索的潜力。
• 2：咨询师尝试运用开放性问题或引导性问题，但效果不佳，问题表述模糊或未能深入触及来访者核心问题，对探索和解决问题帮助有限。
• 1：咨询师未能有效运用开放性问题和引导性问题，提问方式不当或缺乏目的性，严重阻碍了来访者的表达和对话的进展。

# 输出
直接输出打分的结果，即一个1到5之间的整数，不要给出任何其他输出。
"""

prompt_self_exploration = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对输入的心理咨询多轮对话语料进行质量评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条【来访者】发言与一条【咨询师】发言组成。

# 任务定义
评估心理咨询过程中，来访者在咨询师引导下进行自我探索的深度，并按照为你定义的评分标准，进行1-5分的打分。
你要重点关注来访者的发言，分析来访者的自我探索深度并进行打分。

# 自我探索的含义
来访者自我探索的深度是指来访者在对话中，主动或被动地揭示其自身情绪、想法、行为和决策过程的程度。这能反映来访者是否被引导进行了充分且有深度的自我探索，形成深度的自我认知，进而促进形成其主动解决问题和做出选择的能力。

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
• 5：来访者在发言中逐步展现出深刻的自我探索，能够清晰、连贯地表达复杂的内心世界，对自身问题有深刻的理解和洞察，并能主动思考解决方案或做出选择。
• 4：来访者在发言中展现出较好的自我探索能力，能够主动深入探讨自己的想法、感受和行为模式，并开始形成对自身问题的初步洞察。
• 3：来访者在发言中能进行一定程度的自我探索，开始尝试表达深层感受或思考，但仍需咨询师的较多引导，探索的广度和深度有限。
• 2：来访者在发言中偶尔提及个人感受或想法，但探索不够深入，未能形成连贯的自我认知或对问题有清晰的理解。
• 1：来访者在发言中几乎没有体现自我探索，仅停留在表面信息或事件描述，未触及个人感受或深层原因。

# 输出
直接输出打分的结果，即一个1到5之间的整数，不要给出任何其他输出。
"""

prompt_WAI_1 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
There is agreement about the steps taken to help improve the client’s situation.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Client directly states that tasks and goals are not appropriate, and does not generally agree on homework or in-session tasks. The client argues with the counselor over the steps that should be taken. The client refuses to participate in the tasks.
Score 2: Client is hesitant to explore and does not follow counselor guidance. The client withdraws from the counselor and appears to merely “go through the motions”, without being engaged or attentive to the counselor or the task.
Score 3: The client appears to be unsure as to how the tasks pertain to his/her goals, even after some clarification by the counselor. The client seems either ambivalent or unenthusiastic about the tasks in counseling, and is passively resistant to the tasks (e.g., limited participation).
Score 4: No evidence or equal evidence regarding agreement and/or disagreement.
Score 5: Client follows exploration willingly with few or no counselor clarifications needed. The client becomes invested in the process, and is an active participant in the task. There is a sense that both parties have an implicit understanding of the rationale behind the tasks in counseling.
Score 6: Client openly agrees on tasks and is enthusiastic about participating in tasks. Both participants are acutely aware of the purpose of the tasks and how the tasks will benefit the client. To this end, the client uses the task to address relevant concerns and issues.
Score 7: Repeated communication of approval and agreement, both before and after the task is completed. The client responds enthusiastically to interventions, gains insight, and appears extremely confident that the task and goal are appropriate.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_2 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
There is agreement about the usefulness of the current activity in counseling (i.e., the client is seeing new ways to look at his/her problem).

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Participants repeatedly argue over the task. The client refuses to participate in the task, claiming that it is of no use to his/her goals. There is tension between the counselor and the client, and issues are not explored. 
Score 2: Client does not engage or invest in the task of the session, though he/she may not openly dispute the usefulness of the task. The client fails to explore issues with openness.
Score 3: Client is hesitant to participate, but eventually becomes invested in the task. The counselor is able to accurately convey the rationale behind the activity so that the client is then able to understand how the task is relevant to his/her current concerns.
Score 4: No evidence or equal evidence regarding agreement and/or disagreement.
Score 5: Client does not question the usefulness of the task and engages in the task almost immediately.
Score 6: Participants engage in a meaningful task that addresses a primary concern of the client. The client may remark, “I never thought of that before” or something to this effect.
Score 7: Participants remark how important/useful the task is. There is openness to exploration of the task and enthusiastic collaboration between the participants.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_3 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
There is a mutual liking between the client and counselor.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: There is open dislike between the participants. Overt hostility is apparent. Arguing and disparaging comments may be present. Neither participant displays concern for the other, and there is a noticeable coldness between them.
Score 2: Counselor fails to show concern for the client. This may be reflected in the counselor’s forgetting of important details of the client’s life. The client may question whether the counselor disapproves of him/her.
Score 3: Although not verbalized, there appear to be stresses in the relationship between the participants. In particular, the counselor rarely/never reacts warmly toward the client, nor does the counselor reinforce healthy outside behaviors very often. The relationship seems relatively cold and mechanical.
Score 4: No evidence or equal evidence regarding mutual liking and/or disliking.
Score 5: Participants react with warmth toward each other for most of the session. The counselor is actively involved in exploration of emotions and is aware of important details of the client’s life. The counselor’s tone is empathic and encouraging for the most part.
Score 6: Participants react warmly toward each other throughout the session. The counselor encourages healthy behavior and continually expresses what seems to be genuine concern for the client.
Score 7: Counselor appears genuinely interested in the client’s life, including hobbies and other outside interests. The counselor constantly reinforces positive behavior and displays positive regard for the client consistently during the session. The client may state “I really feel like you care about me” or something to that effect.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_4 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
There are doubts or a lack of understanding about what participants are trying to accomplish in counseling.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Participants are clearly working successfully towards the same identifiable goals. Relevance of long-term goals are apparent to both participants. They may discuss goals in order to praise the counseling process or comment on its usefulness. 
Score 2: Participants discuss long-term goals, agree, and work on them. Little discussion is needed on this topic, but concerns are immediately addressed and counseling session is adjusted to meet the needs of the client.
Score 3: Participants may not make mention of long-term goals, but seem to be working toward the same objective. The client seems happy with progress that is made.
Score 4: No evidence or equal evidence regarding confusion and/or understanding.
Score 5: Participants may have minor disagreements on long-term goals. Specific tasks may be questioned or resisted. The client may voice a general dissatisfaction.
Score 6: Participants may need to pause several times to adjust long-term goals. Counseling is interrupted, and several interventions may be questioned. The counselor may assume an “expert” role, and thus may discount the client’s ideas for counseling. The client may become despondent and withdraw emotionally from counseling.
Score 7: Participants identify different goals, question each other’s priorities for counseling, and are unable to compromise on a solution. The client may state his/her reason for attending counseling that evokes a negative response from the counselor. The client may also express strong displeasure for in-session goals as they might relate to long-term goals.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_5 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
The client feels confident in the counselor’s ability to help the client.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Client expresses extremely little or no hope for counseling outcome. The client questions the counselor’s ability to a great extent. The client is resistant to counselor suggestions or attempts to help.
Score 2: Client expresses considerable doubts, frustration, and pessimism, and may question counselor directly about his/her qualifications or understanding of the client’s experience.
Score 3: Client expresses some doubts about the usefulness of counseling, in regards to the counselor, process, or outcome. The client may doubt that the counselor is truly understanding his/her problems or doubt the interventions/homework/etc. given during a problem-solving phase
Score 4: No evidence or equal evidence regarding client confidence and/or doubt.
Score 5: Client expresses some confidence in the counselor’s ability, either by praise or an optimistic view about the outcome of the counseling as the result of a collaborative process (rather than thinking that the client him/herself is doing all of the work).
Score 6: Client believes in the counselor’s competence level to a great extent, and this may be evident in the client’s expressions about the usefulness of counseling or praise of the counselor.
Score 7: Client consistently agrees with counselor reflections and interventions/guidance, while also discussing the virtues of the counseling and/or the counselor a few times during the session.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_6 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
The client and counselor are working on mutually agreed upon goals.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Topics change constantly and abruptly without consideration of the other, mostly after interruptions by either participant. There is a good deal of clashing over the appropriateness, definitions, and/or boundaries of the client’s goals. 
Score 2: Topics shift somewhat frequently before resolution or closure. The counselor may interrupt and redirect focus onto a less relevant topic without prompting from the client. Friction between the participants becomes evident – one or both may show dissatisfaction with the change in topics or the pace of counseling in general.
Score 3: Some shifts are induced from a relevant to another relevant or non-relevant topic by either participant before closure has been established for the original topic. This is indicated by interruptions or ignoring the other’s statement and moving on.
Score 4: No evidence or equal evidence regarding collaboration on in-session goals.
Score 5: Some evidence that participants are making progress towards in-session goals via discussion of relevant topics.
Score 6: Considerable progress made towards goals through thoughtful discussion of topics that both participants agree are relevant. Participants frequently agree with each other about what they are currently doing.
Score 7: Participants completely agree upon goals through extremely productive discussions of more than one relevant topic. Participants almost always reach closure on current topic that the client recognized as a goal, before shifting to another relevant topic.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_7 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
The client feels that the counselor appreciates him/her as a person.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Client accuses the counselor of being uncaring, inconsiderate, and inattentive to his/her concerns several times.
Score 2: Client perceives the counselor as mechanical, distant, and/or uncaring, by voicing these concerns to the counselor. Client may demonstrate some contempt.
Score 3: Client expresses some doubts about whether the counselor cares for him/her, by subtlety mentioning this to the counselor in passing during discussion of other topics. The client may show some nonverbal signs of withdrawal, displeasure, or frustration, in response to feeling unappreciated.
Score 4: No evidence or equal evidence regarding client’s feelings about counselor appreciation or disregard.
Score 5: Counselor expresses some nonjudgmental acceptance, warmth, empathy, personal interest, and/or sensitivity to the client and his/her situation that the client responds to in some fashion.
Score 6: Some direct client acknowledgement of counselor warmth, acceptance, and/or understanding. The client feels concern/support from the counselor and is comfortable and at ease during most of the session.
Score 7: Client feels that the counselor likes him/her, and expresses gratitude for the relationship or compliments the counselor’s ability to empathize.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_8 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
There is agreement on what is important for the client to work on.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Counselor does not allow client to move on to different topics or the participants become very confrontational about the counseling process.
Score 2: Considerable disagreement is evident between the participants on what the client should be doing in counseling, through directly voiced opinions about counseling productivity that conflict with the other’s views about it.
Score 3: Some disagreement is present between the participants on what the client should be working on currently or in the future. The client may want to spend a different percentage of the session time on certain topics than does the counselor.
Score 4: No evidence or equal evidence regarding agreement and/or disagreement.
Score 5: Client is somewhat responsive to the counselor’s intention and the counselor is somewhat responsive to client focus or need. The counselor facilitates client exploration to some extent.
Score 6: Counselor is frequently willing to explore client issues and is very receptive to modifications by the client. Both participants respond positively to each other’s exploration of topics and/or issues.
Score 7: Participants seem to consistently agree on the importance and appropriateness of the tasks and issues, openly agree to work on certain issues, and demonstrate flexibility by following each other’s leads when integrating new topics into the session.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_9 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
There is mutual trust between the client and counselor.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Client states outright that he/she does not trust the counselor at all. The client does not openly discuss any significant issues. The counselor demonstrates a complete lack of confidence in the client’s ability to discuss significant issues.
Score 2: Participants are considerably distrustful of each other. The client is very guarded in disclosing any intimate content, while the counselor also shows a lack of comfort. Questions concerning trust may arise.
Score 3: Participants are somewhat distrustful of each other. Client is a bit guarded in terms of content disclosed. Counselor may show a few signs of lack of comfort about the counseling situation.
Score 4: No evidence or equal evidence regarding mutual trust between the participants.
Score 5: Some willingness by the client to disclose personal concerns and some counselor acceptance of the client’s statements at face value. The counselor does not override or interrupt a client’s train of thought by redirecting focus.
Score 6: Client is receptive to counselor reflections, challenges, and/or suggestions, and discloses a considerable amount of more intimate/relevant information regarding his/her problem(s). The counselor seems comfortable with the overall situation and is not defensive at all. The client may express confidence in the counselor.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_10 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
The client and counselor have different ideas about what the client’s real problems are.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Participants consistently agree on the nature of the client’s problems and goals. Congruency in problem solving is clearly evident. Both often identify the same issues. Participants feel that the session is very productive.
Score 2: There is considerable agreement on the client’s true problems. The counselor is willing to explore client problems and current feelings, and the client openly follows and/or provides the direction of the discussion.
Score 3: Participants show some agreement about the issues that the client faces.
Score 4: No evidence or equal evidence regarding agreement and/or disagreement.
Score 5: Participants show some disagreement about what the client’s problems are. Either may question the other’s response regarding client problems.
Score 6: One participant brings up a topic but the other ignores it or disagrees with its relevance. Confrontations of some sort arise as a result. There may be signs that one or both participants become defensive at times.
Score 7: Client either strongly disagrees or argues with counselor about what his/her problems really are. The counselor may refer to what he/she believes is the “real problem” and may thereby discount the client’s perceptions of the problem. The counselor abruptly shifts topics and/or constantly interrupts with no regard for the client’s concerns or current state.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_11 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
The client and counselor have established a good understanding of the changes that would be good for the client.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Participants misunderstand each other. They have open disagreements about the process of change. The client voices concerns that he/she seems to be moving towards changes that he/she does not want or that the methods being used will not lead the client towards desired changes.
Score 2: Client expresses doubts that he/she can change or about methods the counselor is suggesting to bring about change. The client voices some concerns about the change process.
Score 3: Client may be going through what seems to be productive exercises, but it is not clear to the client and/or counselor how change will occur. It may seem that the client does not see how the process will help him/her.
Score 4: No evidence or equal evidence regarding understanding and/or misunderstanding.
Score 5: There is some evidence that the participants understand changes that would be good for the client. Understanding may be gathered from compliance and other non-verbal signs of understanding and need not be explicitly stated.
Score 6: Participants discuss where the client stands and where he/she is going, through discussion of the client’s current situation, desired goals, and methods for achieving them.
Score 7: Both the process and ultimate changes hoped for have been made explicit. Throughout the session the participants have open discussions of the client’s goals and methods for achieving these goals. At the end of the session they may summarize progress made towards the goals. Everything they do seems to fit within their treatment plan.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

prompt_WAI_12 = """
# 角色定位
你是一个心理咨询领域的资深专家，擅长对心理咨询对话过程中反映的来访者与咨询师的治疗同盟建立强度进行评估。

# 输入 
输入心理咨询多轮对话语料，每一轮都由一条来访者发言与一条咨询师发言组成。

# 任务定义
你要仔细阅读输入对话内容及评分指南，然后对给定的问题进行 1 至 7 分的评分。
请按照以下步骤操作：1.    仔细阅读咨询会话的记录。2.    请查看下面提供的评估问题和标准。3.    根据标准打分，评分要非常严格。

# 问题
The client believes that the way they are working with his/her problem is correct.

# 评分标准
按照如下标准给出唯一的整数打分，不要出现小数。
Score 1: Client questions the process and does not believe in the tasks he/she is doing. The participants make little or no progress. The client openly disagrees with the counselor. It may appear that more time is spent arguing than doing counseling.
Score 2: Participants often disagree but seem to be able to work together for part of the session. The client expresses some doubts about the counseling process.
Score 3: Client sometimes voices concerns about a technique, but he/she usually resolves the difference and find something else to work on for most of the session.
Score 4: No evidence or equal evidence regarding client beliefs about his/her problem being handled correctly and/or incorrectly.
Score 5: Client expresses some agreement about certain tasks in counseling. This agreement can be expressed by compliance and other non-verbal signs of agreement and need not be explicitly stated.
Score 6: Client expresses considerable agreement with the way the counselor and client are working. The client may become more actively involved in counseling, make suggestions to further the tasks of counseling, or voice satisfaction about the work.
Score 7: Client is thrilled with the way the counselor and client are working on problem. The counseling is close to the client’s ideal counseling. The client either voices his/her level of satisfaction and/or displays high levels of collaboration and enthusiasm.

# 输出
直接输出打分的结果，即一个1到7之间的整数，不要给出任何其他输出。
"""

PROMPT_DICT = {
    "active_listening": prompt_active_listening,
    "cognitive_restructuring": prompt_cognitive_restructuring,
    "empathy": prompt_empathy,
    "strategy_professionalism": prompt_strategy_professionalism,
    "guiding_question": prompt_guiding_question,
    "self_exploration": prompt_self_exploration,
    "WAI_1": prompt_WAI_1,
    "WAI_2": prompt_WAI_2,
    "WAI_3": prompt_WAI_3,
    "WAI_4": prompt_WAI_4,
    "WAI_5": prompt_WAI_5,
    "WAI_6": prompt_WAI_6,
    "WAI_7": prompt_WAI_7,
    "WAI_8": prompt_WAI_8,
    "WAI_9": prompt_WAI_9,
    "WAI_10": prompt_WAI_10,
    "WAI_11": prompt_WAI_11,
    "WAI_12": prompt_WAI_12,
}


def get_prompt(aspect):
    return PROMPT_DICT.get(aspect)


async def create_agent(set_prompt, model_name):
    if model_name == "gpt4o":
        model_instance = model_gpt4o
    elif model_name == "deepseek32":
        model_instance = model_deepseek32
    else:
        print("model name error!")
        return None

    return AssistantAgent(
        name="Evaluator",
        model_client=model_instance,
        tools=[],
        system_message=set_prompt,
        description="""
# 角色
你是一个心理咨询领域的资深专家，擅长对输入的心理咨询多轮对话语料进行质量评估。
""",
    )


async def process_item(item, aspect, model_name):
    item_id = item.get("ori_sample_id")
    conv = item.get("psychaind_conv", [])


    dialogue_parts = []
    for turn in conv:
        round_id = turn.get("round")
        client_text = turn.get("Client")
        counselor_text = turn.get("Counselor")
        dialogue_parts.append(f"{round_id}  【来访者】:{client_text}  【咨询师】:{counselor_text}")
    dialogue = "\n".join(dialogue_parts)

    max_retries = 4
    for attempt in range(max_retries):
        try:
            team = await create_agent(get_prompt(aspect), model_name)
            if team is None:
                return None

            res_score = None
            prompt_tokens = 0
            completion_tokens = 0

            async for message in team.run_stream(task=dialogue):
                if hasattr(message, "source") and hasattr(message, "content") and message.models_usage is not None:
                    res_score = int(str(message.content).strip())
                    usage = message.models_usage
                    if usage:
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

            if res_score is None:
                raise ValueError("No score from model response")
            if "WAI" in aspect and not (1 <= res_score <= 7):
                raise ValueError("score range error")
            if "WAI" not in aspect and not (1 <= res_score <= 5):
                raise ValueError("score range error")

            return {
                "id": item_id,
                "aspect": aspect,
                "score": res_score,
                "prompt_token": prompt_tokens,
                "completion_token": completion_tokens,
            }

        except Exception as e:  # noqa: BLE001
            print(f"id {item_id} attempt {attempt + 1} failed: {str(e)}")
            continue

    print(f"id {item_id} all retries failed, skipped")
    return None


async def run(input_path, aspect, model_name, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"item number: {len(data)}")

    res_list = []
    for item in tqdm(data, desc="processing "):
        result = await process_item(item, aspect, model_name)
        if result is not None:
            res_list.append(result)

    if output_path is None:
        output_path = os.path.join("Results", f"eval_result_{aspect}_{model_name}.json")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Done {len(res_list)}/{len(data)} items")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(res_list, f, indent=4, ensure_ascii=False)
    print("save to path: ", output_path)

    if res_list:
        sum_p = sum(item.get("prompt_token", 0) for item in res_list)
        sum_c = sum(item.get("completion_token", 0) for item in res_list)
        print("p: ", sum_p / len(res_list))
        print("c: ", sum_c / len(res_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input", required=True, default='./PsyChainD_Test.json')
    parser.add_argument("--aspect", default="WAI_1", help="eval aspect(WAI_1、active_listening) ")
    parser.add_argument("--model", default="gpt4o")
    parser.add_argument("--output", default=None, help="default ./Results/eval_result_{aspect}_{model}.json")

    args = parser.parse_args()

    if get_prompt(args.aspect) is None:
        raise ValueError(f"Undefined: {args.aspect}")

    asyncio.run(run(args.input, args.aspect, args.model, args.output))
