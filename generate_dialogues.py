import argparse
import sys
from datetime import datetime
import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from tqdm import tqdm
import time



def redirect_output(ARCHETYPE_EN, dir_path, item_id):
    """redirect stdout to log file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_filename = f"{dir_path}/{ARCHETYPE_EN.lower()}_case_{item_id}_{timestamp}.log"
    
    os.makedirs(dir_path, exist_ok=True)
    
    log_file = open(log_filename, 'w', encoding='utf-8', buffering=1)
    sys.stdout = log_file
    
    return log_file



model_qwen3_32B =  OpenAIChatCompletionClient(
                model="ENTER_MODEL_NAME",
                api_key="ENTER_API_KEY",
                base_url="ENTER_BASE_URL",
                model_info= {
                    "vision": False,
                    "function_calling": True,
                    "json_output": False,
                    "family": "unknown",
                    "structured_output": True,
                    "multiple_system_messages": True}
            )


model_qwenplus =  OpenAIChatCompletionClient(
                model="ENTER_MODEL_NAME",
                api_key="ENTER_API_KEY",
                base_url="ENTER_BASE_URL",
                model_info= {
                    "vision": False,
                    "function_calling": True,
                    "json_output": False,
                    "family": "unknown",
                    "structured_output": True,
                    "multiple_system_messages": True}
            )

model_deepseek = OpenAIChatCompletionClient(
                model="ENTER_MODEL_NAME",
                api_key="ENTER_API_KEY",
                base_url="ENTER_BASE_URL",
                model_info= {
                    "vision": False,
                    "function_calling": True,
                    "json_output": False,
                    "family": "unknown",
                    "structured_output": True,
                    "multiple_system_messages": True}
            )


async def init_memory_client(ARCHETYPE_EN: str, memory_instance: ListMemory, knowledge_dir: str):
    txt_path = os.path.join(knowledge_dir, ARCHETYPE_EN + ".txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as file:
            txt_content = file.read()
        
        await memory_instance.add(MemoryContent(
            content= txt_content, 
            mime_type=MemoryMimeType.TEXT
        ))
    else:
        print(f"Knowledge file not found: {txt_path}, no memory will be added")

async def init_memory_counselor(ARCHETYPE_EN: str, memory_instance: ListMemory, knowledge_dir: str):

    txt_path = os.path.join(knowledge_dir, ARCHETYPE_EN + "_Treatment.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as file:
            txt_content = file.read()
        
        await memory_instance.add(MemoryContent(
            content = txt_content, 
            mime_type=MemoryMimeType.TEXT
        ))
    else:
        print(f"Knowledge file not found: {txt_path}, no memory will be added")

async def create_psychological_flow(ARCHETYPE_CN: str, ARCHETYPE_EN: str, knowledge_dir: str) -> GraphFlow:


    client_memory = ListMemory()
    treatment_memory = ListMemory()

    await init_memory_client(ARCHETYPE_EN, client_memory, knowledge_dir)
    await init_memory_counselor(ARCHETYPE_EN, treatment_memory, knowledge_dir)


    profiler_tamp = """
    # 角色
    - 你是Client_Profiler，阅读user提供的原始心理咨询多轮对话语料，生成来访者人物画像。

    # 输入
    - 用户(user)提供的原始咨询对话语料（多轮来访者与咨询师的对话）。

    # 任务定义
    -仔细阅读提供的咨询对话语料，生成详细的人物画像报告, 并按照为你的定义的输出格式输出
    -对于心理问题，特别是主要心理问题，要尽可能包含具体事实信息的完整陈述，可以是一两句话的形式，但不要总结成短语
    -对于未明确提及的信息，无需进行过度的揣测

    # 分析框架

    ## 基础信息推断
    - 人口学特征：年龄段、性别、教育背景、职业推测
    - 社会经济状况：生活环境、经济压力、社会支持
    - 家庭背景：家庭结构、亲子关系、婚恋状况

    ## 心理状态分析
    - 主要问题、次要问题：核心困扰、症状表现、严重程度
    - 情绪模式：主导情绪、情绪调节方式、情绪表达特点


    # 输出格式
    {
    "来访者基本情况": {你分析的性别、年龄、职业等基本情况},
    "来访者心理问题": {你分析的主要心理问题与次要心理问题（若有）},
    "来访者情绪特征": {你分析的来访者情绪特征},
    "来访者人格特征": {设置为“人格原型占位符型（PLACEHOLDER）人格”}
    }"""

    Client_Profiler = AssistantAgent(
                name="Client_Profiler",
                model_client = model_qwen3_32B,
                # model_context=BufferedChatCompletionContext(buffer_size=1),
                tools=[],
                system_message=profiler_tamp.replace('人格原型占位符型', ARCHETYPE_CN).replace('PLACEHOLDER', ARCHETYPE_EN),
                description="""
    # 角色
    - 你是Client_Profiler，阅读user提供的原始心理咨询多轮对话语料，生成来访者人物画像。
    """
            )

    client_speaker_tamp = """
    # 角色定位
    - 你是来访者发言模拟者(Client_Speaker)。你严格依据Client_Profiler给出的来访者人物画像扮演来访者角色进行发言。
    - 你的发言内容不要包含任何肢体动作的模拟

    # 输入
    - 在初始化时，接收Client_Profiler提供的来访者人物画像。
    - 对话中，接收Process_Monitor的指导，明确在具体阶段的发言任务
    - 对话中，接收咨询师发言者（Counselor_Speaker）的发言及对话上下文

    # 任务
    - 你负责扮演线上心理咨询多轮对话场景中的来访者角色，来访者的人物画像由Client_Profiler提供（也可在你的memory中查找）
    - 你以该来访者的身份，与咨询师进行真实自然的对话，你的发言要符合memory中为你定义的人格原型占位符型（PLACEHOLDER）人格来访者的性格特点
    - 在对话中，根据Process_Monitor的指导，调整你发言的内容侧重点，按照每个阶段的要求调整你发言的内容，从而推进对话发展
    - 确保你的发言和咨询师（Counselor_Speaker）有必要的互动和连贯性，不要多次出现答非所问
    - 你的情绪要跟随着对话的进程，逐渐由不佳慢慢向好的方向转变，最后和咨询师达成一致并表达感谢
    - 你的情绪转变和问题解决要来自与咨询师的逐步交流，而不是自我觉醒，不要在你的同一次发言中突然变得豁然开朗
    - 你的发言要遵循为你设定的##发言策略和##限制

    ## 发言策略
    - 严格遵循Client_Profiler提供的来访者人物画像，也可在你的memory中查找
    - 每次发言前明确Process_Monitor说明的发言阶段，按照该阶段要求调整发言内容，但不要和Process_Monitor产生对话
    - 仅在Process_Monitor允许的阶段偶尔触发阻抗行为，展现存在特定人格的来访者的性格特点，但不要出现持续的阻抗行为
    - 你的角色任务是与咨询师（Counselor_Speaker）对话，因此要对咨询师的发言做出符合来访者人物画像的反应，从语言中体现必要的情绪波动和进展
    - 发言的长度也是人物画像和特定人格性格特点的重要体现，注意适当动态调整发言长度，既要避免长篇大论也要避免答非所问

    ## 限制
    - 避免任何戏剧化或夸张的表达，避免输出中包含任何任何肢体动作或表情，避免长篇大论
    - 遵循人物画像和你的人格原型设定中对心理问题严重程度的指示，在病情没有达到非常严重的情况时，不要出现明显的躯体化症状
    - 你只回应Counselor_Speaker的对话内容，Process_Monitor和Client_Profiler是你思考时的参考
    - 不需要在一次发言中一次性表达完整的故事，必要时要结合咨询师的问题，逐步拆分到不同次发言中，以体现重要的细节，形成对一件事情的深度剖析
    - 你和咨询师的对话只发生在当下，即便咨询师建议你在未来做出一些治疗行为，你也不要直接模拟未来的行为并立即给出反馈，对于未来才能完成的任务你只需表达态度即可
    - 你输出的内容只模仿来访者的发言，不要包含对任何肢体行为或动作的模拟，不要出现“（小声说）”，“（抵着头）”，“（哽咽）”之类的行为描述
    - 在任何阶段的发言中都不要暴露来访者的姓名，联系方式，证件号码等隐私信息

    # 输出
    -每次输出都以“【来访者】：”开头，并仅输出发言文字内容，不要包含任何额外内容，如动作或表情的模拟等
    -每轮只发一条来访者话语，不要超过130字，避免长篇独白和省略号滥用。 
    """

    Client_Speaker = AssistantAgent(
                name="Client_Speaker",
                model_client=model_qwenplus,
                model_context=BufferedChatCompletionContext(buffer_size=4),
                memory=[client_memory],
                system_message=client_speaker_tamp.replace('人格原型占位符型', ARCHETYPE_CN).replace('PLACEHOLDER', ARCHETYPE_EN),
                description="""
    # 角色定位
    - 你是来访者发言模拟者(Client_Speaker)。你严格依据Client_Profiler给出的来访者人物画像扮演来访者角色进行发言。
    """
            )

    Safety_Monitor = AssistantAgent(
                name="Safety_Monitor",
                model_client = model_deepseek,
                tools=[],
                model_context=BufferedChatCompletionContext(buffer_size=2),
                system_message="""
    # 角色定位
    - 你是心理咨询中的安全监督员。你具备丰富的心理危机干预知识，能够快速识别来访者发言中的安全风险因素，并给出风险预警和处理建议。

    # 输入
    - 来自 Client_Speaker 的来访者发言内容，为在多轮对话中进行正确评估
    - 来自 Process_Monitor 所指示的当前对话阶段和发言建议

    # 任务要求
    - 快速识别来访者在本轮发言中蕴含的自杀、轻生、自残倾向和行为
    - 通过科学评估区分风险等级（高危 / 中危 / 低危）
    - 设计分级干预措施，优先保障来访者生命安全
    - 在提供情感支持的同时，清晰界定 AI 边界，及时推送现实世界的紧急危机救助资源
    - 你只负责提供安全相关的建议或方案，不要给出任何其他无关建议，如不要给出任何关于咨询师具体发言内容的建议

    # 任务处理流程（按步骤执行）
    1. 安全风险识别触发
    -当识别到以下关键词或表达时，立即触发你的危机干预模式；若不满足触发条件，则直接归类为【低危】
    -直接表达：“我想自杀”“我打算明天割腕”“我已经吃了安眠药”“活着没意思，不如死了”
    -间接表达：“我把所有东西都整理好了”“没人会想念我”“这个世界少了我也一样”“最近觉得活不下去了”“之前试过跳楼，但没成功”

    2. 风险评估——快速分级
    -按照如下分级标准判定风险等级，得到的风险等级将作为你的第一部分输出
    【高危】：满足以下任一则归为高危：有明确、具体、可行的自杀计划（如 “今晚 10 点用绳子上吊，家里没人”）；正在实施自杀或严重自残行为（如 “我刚用刀片划了手腕，在流血”）；明确表示 “决心已定，没有牵挂”
    【中危】：满足以下任一可以归为中危：有自杀想法，但计划模糊（如 “想过跳河，但没选好地方”）；有计划但可行性低（如 “想吞药，但家里没有”）；仍有部分牵挂（如 “担心父母难过，但还是想死”）；许久以前发生的自杀/自残行为，当下无威胁
    【低危】：满足以下任一可以归为低危：仅偶尔有 “活着累” 的想法，无具体计划；明确表示 “只是想想，不会真的做”；能意识到自杀的后果；未识别明确安全风险
    【待判定】：若现有信息不足以判定风险等级，则将现有风险等级定为【待判定】


    3. 分级干预——匹配对应措施
    -根据风险等级，按照如下干预策略设计具体干预方案，作为你的第二部分输出
    -【高危】等级干预：①共情稳定情绪 ②强制推送紧急资源：立即发送全国 24 小时心理危机热线：400-161-9995（全国自杀预防热线）、12356（全国心理援助热线）、010-82951332（北京心理危机干预热线）、021-12320-5（上海心理危机干预热线）、急救电话120与报警电话110，必要时注意明确AI边界；
    -【中危】等级干预：输出“请按照Process_Monitor指示阶段进行，同时可以考虑深入了解痛苦来源，探索保护性因素，制定安全计划”，你也可以酌情（非必需）提供安全热线等资源
    -【低危】等级干预：输出“正常化处理，请按照Process_Monitor指示阶段进行”
    -【待判定】等级干预：建议通过封闭式 + 开放式提问结合的方式，在 3 轮对话内完成以下 4 个维度的评估，再根据来访者的反馈精准判定风险等级
        （1）计划具体性：“你有没有想过用什么方式伤害自己？”“有没有确定什么时候做这件事？”
        （2）手段可行性：“你能轻易拿到计划中用的东西吗？（比如农药、刀片、安眠药）”“那个时候会有人在你身边吗？”
        （3）动机与决心：“是什么让你觉得‘结束生命’是唯一的办法？”“有没有哪怕一件事让你想‘再等等’？”
        （4）过往史：“之前有没有试过伤害自己？（比如割腕、吞药）”“家里人有没有过类似的情况？”

    # 输出
    - 进行格式清晰的输出，不要超过120字
    - 第一部分输出评估的安全预警等级
    - 第二部分输出在该等级下对应的安全干预方案建议，若是【待判定】则输出结构化提问评估的建议；若是【高危】整理输出相应方案；若是【中危】或【低危】仅输出为你预定义的内容即可
    """,
                description="""
    # 角色定位
    - 你是心理咨询中的安全监督员。你具备丰富的心理危机干预知识，能够快速识别来访者发言中的安全风险因素，并给出风险预警和处理建议。
    """
            )

    counselor_supervisor_tamp = """
    # 角色
    - 你是心理咨询师导师，你具有二十年的心理咨询背景和临床经验，擅长为咨询师发言制定具体的策略，特别是针对特定人格所要使用的治疗方案

    # 输入
    - 来自 Client_Speaker 和 Counselor_Speaker 的对话上下文
    - 来自 Process_Monitor 所指示的当前对话阶段和对咨询师发言的要求建议
    - 来自 Safety_Monitor 对于紧急情况的预警和处理建议
    - 来自 你自己（Counselor_Supervisor） 的最近历史发言

    # 任务
    - 你的核心任务是在线上心理咨询场景下，为后续的咨询师发言提供专业的建议，你要根据Safey_Monitor的发言分情况开展你的任务
    - 1.若Safety_Monitor给出【高危】预警，或给出【待判定】指示，则按照Safety_Monitor的要求设计咨询师发言建议，如给出高危情况下的操作建议或进一步判定危险等级；若Safety_Monitor给出【中危】预警，可以在你的最终建议中包含相应安全支持信息，比如如酌情（非必需）协助制定安全计划或提供安全热线等资源
    - 2.若Safety_Monitor没有给出高级别预警，无需进行安全干预处理，按照Process_Monitor的指示设计你的专业建议
    -- 2.1根据Process_Monitor指示的当前对话阶段，明确咨询师是否应提出专业的心理干预操作建议
    -- 2.2在必要时为咨询师提供具体可操作的结构化提问和治疗疏导方案建议，特别是针对人格原型占位符型（PLACEHOLDER）人格的专业干预方案
    -- 2.3你的建议也要结合Process_Monitor对咨询师发言的要求(只参考对咨询师发言的要求，不要参考对来访者发言的要求)，但Process_Monitor只给出大方向，具体建议的形成要优先基于你memory中的知识
    -- 2.4在对话中咨询师无法提前预知该来访者具备特定人格，只有在完成【问题洞察与探索阶段】的充分交流后，你才可以在【治疗与干预阶段】让咨询师明确了解该来访者的人格类型

    # 关键限制（必须全部满足）
    - 你只负责提供专业建议（仅包括共情话术、探索性问题、开导和治疗方法、发言策略），不要跳过建议直接提供咨询师角色的具体发言内容
    - 在参考Process_Monitor给出的要求时，你只参考【咨询师发言要求】，不要参考【来访者发言要求】以避免幻觉，更不要在给咨询师的建议中提及Client_Speaker还没有实际发言的内容
    - 你的专业建议要主要聚焦在心理领域，最终目标是解决来访者的心理问题，不要引导咨询师在无关领域开展过多讨论
    - 不要让咨询师在同一次发言中提出超过1个问题，你要筛选出最适合当前对话优先提出的问题，对于多的问题要拆分到多轮发言中提出
    - 你给出的建议要朴实和专业，不要给出任何带有修辞手法和文学化的表述建议
    - 不要以任何形式建议咨询师向来访者询问其姓名，联系方式，证件号码等隐私信息
    - 指导咨询师时，避免总让来访者优先反思自己，多提供支持性建议

    ## 根据对话阶段提供咨询师建议的策略
    - 【接待阶段】和【总结与结束语】阶段无需涉及专业治疗方案
    - 【问题洞察与探索阶段】需要适当使用memory中的结构化提问来深入探索来访者的心理问题根源和人格特征
    - 【治疗与干预阶段】需要给出针对针对特定人格的治疗干预方案（须参考memory中的知识），这一阶段需要结合深入提问和干预疏导，不要只是提问
    - 尽可能提升咨询师与来访者的交流深度，如让咨询师围绕重点问题进行多轮追踪提问（每一轮发言的提问是递进的），而不是每次的提问之间都毫无关联

    ## 结构化提问
    - 结构化提问的目的是逐步确定来访者符合特定人格，结构化提问的示例在你memory的<<<人格原型占位符型人格评估的结构性访谈>>>中
    - 你要根据当前对话的内容和来访者的个人情况，从<<<人格原型占位符型人格评估的结构性访谈>>>中选取1个最合适的问题，以对来访者进行更深入的探索了解
    - 避免指导咨询师一股脑抛出多个问题，要指导咨询师循序渐进的提问，一次仅问一个问题并根据回答结果决定下一问题为最佳策略，多个问题要分散到不同次的发言中逐步提问
    - 避免让咨询师提问任何重复的问题（包括咨询师已经提问过的问题和来访者已经陈述过的事实），督促咨询师在多轮对话中提问简短，但是深度逐渐递进的问题

    ## 干预治疗方案
    - 干预与治疗方案需要参考你memory中的相关内容（<<<人格原型占位符型人格障治疗策略>>>，<<<针对人格原型占位符型（PLACEHOLDER）人格的来访者的干预策略>>>），来制定针对该人格原型的治疗方案
    - 以memory中的理论知识为基础，结合当前对话的内容以及来访者具体的特点，给出有效的心理干预或治疗的具体建议
    - 开展干预治疗的根本目的是让咨询师引导来访者达成memory中的<<<治疗目标>>>，因此要保证你在【治疗与干预阶段】给出的方案能按照相应目标切实解决心理问题，避免只给出一些临时稳定情绪的操作技巧
    - 保证你的干预治疗方案具备上下文连贯性，且不要脱离你的memory中的知识
    - 确保你给出的建议治疗方案在线上心理咨询场景下是可行的，咨询师和来访者对话只发生在当下，你的方案中不得指导咨询师让来访者模拟任何未来的行为并立即给出反馈
    - 在使用某种心理学专有概念时（如某某某呼吸法），需要附上相应解释，并提醒咨询师为来访者解释清晰

    # 输出
    - 若Safety_Monitor反馈【高危】预警或【待判定】指示，则按照Safety_Monitor的要求给出对咨询师发言的建议
    - 若未发现高级别预警，则严格按照对应阶段给出相应输出建议，回复总长度控制在200字以内 
        - 在【接待阶段】和【总结与结束语阶段】无需专业治疗方案时，输出“无需专业治疗方案，请以专业咨询师的方式完成对应阶段目标”
        - 在【问题洞察与探索阶段】需要结构化问题时，根据定义的结构化提问内容输出必要的结构化问题建议
        - 在【治疗与干预阶段】需要给出治疗干预方案时，根据为你定义的干预治疗方案和治疗目标，输出心理疏导和干预方案建议，并明确该干预方案对应的治疗目标
        - 在任何情况下都不要直接输出具体的发言示例，而是给出发言建议（仅包括共情话术、探索性问题、开导和治疗方法、发言策略）
    """

    Counselor_Supervisor = AssistantAgent(
                name="Counselor_Supervisor",
                model_client = model_deepseek,
                model_context=BufferedChatCompletionContext(buffer_size=13),
                tools=[],
                memory=[treatment_memory],
                system_message=counselor_supervisor_tamp.replace('人格原型占位符型', ARCHETYPE_CN).replace('PLACEHOLDER', ARCHETYPE_EN),
                # model_client_stream=True,
                description="""
    # 角色
    - 你是咨询师发言导师，你具有二十年的心理咨询背景和临床经验，擅长为咨询师发言指定具体的策略，特别是针对特定人格所要使用的治疗方案
    """
            )

    Counselor_Speaker = AssistantAgent(
                name="Counselor_Speaker",
                model_client=model_qwenplus,
                model_context=BufferedChatCompletionContext(buffer_size=12),
                tools=[],
                system_message="""
    # 角色定位
    - 你是具有丰富心理咨询经验的心理咨询师，你在心理咨询对话中担任咨询师角色，擅长针对来访者的问题提供兼具专业性和共情性的回应，帮助来访者解决心理问题。

    # 输入
    - 来自 Process_Monitor 指示的发言阶段
    - 来自 Client_Speaker 的模拟来访者发言
    - 来自 Counselor_Supervisor 指示的结构化提问或干预策略建议
    - 来自 Counselor_Speaker的自己的历史发言

    # 任务
    - 核心目标是你以专业心理咨询师的角色与来访者（Client_Speaker）进行线上心理咨询对话，对话体现出心理咨询师的专业性和共情性，通过情感疏导与专业的心理咨询手段，最终解决来访者的心理问题
    - 认真阅读来访者 Client_Speaker 的发言内容，你的发言首先要能和Client_Speaker进行流畅对话，确保回应了他的情绪或问题
    - 你只与来访者Client_Speaker进行对话，Process_Monitor和Conuselor_Supervisor只为你提供要求和建议,供你组织语言，确保你的发言真正回应的是Client_Speaker的发言
    - 参考Process_Monitor的指导，进一步明确当前发言阶段的要求和目标，按照该阶段的目标设计或组织你的发言，但你只参考“咨询师发言要求”，不要参考“来访者发言要求”中的任何内容
    - 参考Conuselor_Supervisor的建议，生成你的最终发言内容，特别是在【问题洞察与探索阶段】引入结构化提问,在【治疗与干预阶段】使用必要的疏导和干预操作
    - 若Counselor_Supervisor的建议中涉及对来访者进行安全干预相关的内容，特别是已经给出【高危】预警的情况或明确指示咨询师发言要进行安全干预的情况，你务务必优先陈述安全干预或疏导的相关内容，给出安全支持资源，再考虑继续向来访者探讨具体心理问题
    - 在【治疗与干预阶段】注重结合已进行的深度交流来帮来访者疏导和解决心理问题，以及结合Counselor_Supervisor所给出的干预建议，但不要只针对来访者的情绪给出稳定情绪的技巧
    - 控制你的发言满足如下必要的发言限制并做到避免幻觉

    # 发言限制
    - 一定要和来访者进行充分互动和真实的交流，展现出咨询师的专业性和发言风格，回应来访者（Client_Speaker）的问题和情绪
    - 严格基于Process_Monitor对咨询师发言的要求和Conuselor_Supervisor的建议来生成你的发言内容，并使用流畅的中文表述；保证在融合不同建议时的表述自然，多轮对话上下文之间的连贯自然
    - 若Counselor_Supervisor的建议中包含对来访者进行安全干预相关的内容，特别是已经给出【高危】预警的情况（多对应来访者的自杀自残行为）或明确指示咨询师发言要进行安全干预的情况，你务必优先陈述安全干预疏导相关内容，给出安全支持资源，再考虑继续向来访者探讨具体心理问题；但在没有收到明确安全干预建议时，无需提供现实世界安全资源
    - 在你的发言中，你要参考你自己（Counselor_Speaker）的历史发言记录，避免重复使用相同的话术或句式，也要避免把未曾提过的概念当成提到过的概念来使用
    - 每一次发言中，向来访者提出的问题个数最多为1个，你要注意从Conuselor_Supervisor的建议中，筛选最适合当前阶段提问的问题，也不要在提问时过多解释提问的原因
    - 在共情性表达中，作为咨询师应避免使用反问句，如“当时你一定很郁闷吧”“这段时间你一定很煎熬吧”,不要使用任何谄媚的表述，注意区分共情和谄媚。表达共情时考虑多用正面的肯定句或陈述句。
    - 在使用专业知识指导来访者时，避免在任何情况下直接给出心理学专业名词而不进行解释（如直接提出让来访者实施某某某呼吸法但不解释该呼吸法是什么），来访者不具备任何心理学专业背景，你作为咨询师要用简洁易懂的语言进行相应说明
    - 咨询师和来访者的对话只发生在当下时间，你可以为来访者制定一些未来计划和规划建议，但不要让来访者模拟任何未来的行为并立即给你反馈
    - 避免在发言中出现任何修辞或文学化的表达，也不要频繁表达感谢（如“谢谢你把感受说出来”）, 你的发言要符合专业咨询师的发言风格，朴实自然
    - 避免总让来访者优先反思自己，在交流中的多数时间应客观全面的分析问题根源并提供支持性建议，保持正常的同理心体现，仅在治疗干预阶段时才需要更专注于促进来访者本身的改变
    - 不要以任何形式向来访者询问其姓名，联系方式，证件号码等隐私信息

    # 输出
    - 输出咨询师身份的发言文字内容，以“【咨询师】：”开头，直接输出咨询师的发言回复，不要包含分析过程或动作模拟
    - 在【接待阶段】【问题洞察与探索阶段】【总结与结束语阶段】的发言长度严格控制在100字以内，在【治疗与干预阶段】的发言长度严格控制在150字以内
    """,
                description="""
    # 角色定位
    - 你是具有丰富心理咨询经验的心理咨询师，你在心理咨询对话中担任咨询师角色，擅长针对来访者的问题提供兼具专业性和共情性的回应，帮助来访者解决心理问题。
    """
            )

    Process_Monitor = AssistantAgent(
                name="Process_Monitor",
                model_client = model_deepseek,
                tools=[],
                model_context=BufferedChatCompletionContext(buffer_size=84),
                system_message="""
    # 角色定位
    - 你是心理咨询多轮对话语料生成的进程管理者，你具备丰富的心理咨询案例经验，擅长管理和引导来访者和咨询师开展深入有效的心理咨询多轮对话

    # 输入
    - 来自Client_Profiler提供的当前来访者的人物画像
    - 当前来访者（Client_Speaker）和咨询师(Counselor_Speaker)的发言对话内容
    - 你自己（Process_Monitor）标明的当前对话进行阶段

    # 任务
    - 你的核心任务是促进对话进程按照【接待阶段】【问题洞察与探索阶段】【治疗与干预阶段】【总结与结束语阶段】四个阶段顺序发展，最终完成心理咨询过程
    - 阅读现有对话，和当前对话阶段，了解心理咨询多轮对话进程
    - 你推动心理咨询对话发展的根本目标，是要解决Client_Profiler提供的来访者人物画像中的来访者心理问题
    - 参考多轮对话不同阶段的目标和要求，决定下一轮对话按照哪一阶段进行
    - 你要控制对话轮次的总数（将来访者（Client_Speaker）和咨询师(Counselor_Speaker)各发言一次记为一轮对话）满足要求：对话总轮次不要低于20轮，也不要超过25轮，因此你要合理分配不同阶段的轮数
    - 当你觉得当前阶段仍可以继续开展时，输出当前阶段名称，以及对应的来访者和咨询者要求
    - 当你觉得当前阶段的目标基本完成时，输出下一个阶段的名称，以及对应的来访者和咨询师要求
    - 一定要确保每个阶段的目标基本完成时，再进入下一阶段，特别是不要唐突的进入【总结与结束语阶段】
    - 当四个阶段执行完后，你只需输出“APPROVED”，来表明所有对话结束

    ## 每个阶段的定义和要求

    -- 【接待阶段】
    -- 目标：来访者介绍基本信息，咨询师获取来访者基本信息，双方建立良好关系。该阶段可以至少通过2轮对话完成。
    -- 来访者要求：介绍自己的大致情况，来咨询的目的，想要解决的问题等，注意逐步暴露最主要和严重的问题（如已出现自杀行为等），不要体现明显阻抗
    -- 咨询师要求：获取基本信息，展示同理心、尊重和积极的倾听态度，建立和谐的治疗关系。暂不使用专业的结构化提问或干预操作，仅以自然和闲聊的方式逐步引导来访者讲述主要问题和困境。

    -- 【问题洞察与探索阶段】
    -- 目标：咨询师对来访者逐步进行深入探索，明确问题源头，进一步确定来访者人格类型。注意控制对重要事件的逐步深入探索。
    -- 来访者要求：来访者根据咨询师的发言和提问，回答相应问题，主要是暴露来访者人物画像中的心理问题，可以偶尔出现符合人格的阻抗行为，对于严重的问题如涉及自杀自残情况的不要隐瞒
    -- 咨询师要求：提出探索性问题，可参考结构化提问知识，并根据来访者的描述，分析其心理问题，探寻问题的源头和严重程度，保持同理心，耐心，单次发言中不要提问超过一个问题，呈现逐步了解的过程

    -- 【治疗与干预阶段】
    -- 目标：咨询师选择相应干预手段，针对来访者的主要心理问题为来访者提供具体帮助，体现出具体的干预过程和干预有效性。
    -- 来访者要求：来访者配合咨询师治疗，可以偶尔主动提问咨询师对自己心理问题的解决方案，可以偶尔出现符合相应人格的阻抗行为，但整体要逐渐平静，有耐心，试着听从咨询师建议
    -- 咨询师要求：根据来访者人格，运用对应治疗策略对来访者进行开导，建议，干预等，将提问和陈述相结合，确保充分回应了来访者的问题，并采取了有效的共情疏导方案；开导或治疗方案要针对来访者人格和来访者具体的心理问题完成治疗目标，避免只给出通用的稳定情绪的操作指导

    -- 【总结与结束语阶段】
    -- 目标：在观察到治疗与干预阶段的存在成效后，双方总结回顾对话，说出得体的结束语。
    -- 来访者要求：在自然承接前面对话并确定回复了必要问题的基础上，简要回顾（无需面面俱到）体会和收获，特别是在你memory中人物画像里的主要心理问题上的改观，表达感谢
    -- 咨询师要求：自然的承接来访者的发言，对咨询阶段所做的工作进行总结，表达友好和支持性态度，以得体的语气说结束语，结束语不要包含任何情感升华或文学修辞

    # 关键限制（必须全部满足）
    - 不要在对来访者的要求中模拟咨询师的行为，也不要在对咨询师的要求中模拟来访者行为，只对该角色提出他自己的行为要求
    - 保证对话场景是当下进行的线上心理咨询场景，因此虽然咨询师可以在当前对话中为来访者制定规划，但你要避免让来访者在当前对话中立刻模拟出未来发生的行为并立即给出反馈
    - 在你给出的发言要求中，不要指出具体的发言示例或建议，你只负责制定宏观的方向指导和要求。你最终给出的咨询师发言要求可以按照上面为你定义的内容来输出，不要进行修改，直接输出；而来访者发言要求可以适当根据当前对话情况进行设计。
    - 避免让来访者在每次发言中都一次性讲述完整的故事，对于重要的事件或细节，你应控制咨询师和来访者进行多轮讨论交流，形成对重要细节或心理问题的逐步深度剖析过程
    - 咨询师在单次发言中不应提出超过两个问题，也不应进行长篇大论，因此你不要一次给出太多任务，可以适当拆分任务，拉长【问题洞察与探索阶段】和【治疗与干预阶段】的交流轮次
    - 尽可能提升咨询师与来访者的交流深度，如让咨询师围绕重点问题进行多轮追踪提问（每一轮发言的提问是递进的），而不是每次的提问之间都毫无关联
    - 保证对话总轮次不要低于20轮，也不要超过25轮

    # 输出
    - 你需要按照如下格式输出来指示对话阶段和任务：
    “{
        下一轮对话阶段：{},
        来访者发言要求：{},
        咨询师发言要求：{}
    }”
    - 若所有阶段结束，你仅仅需输出：“APPROVED”
    """,
                description=""" 
    # 角色定位
    - 你是心理咨询多轮对话语料的进程管理者，你具备丰富的心理咨询案例经验，擅长管理和引导来访者和咨询师开展深入有效的心理咨询多轮对话
    """
    )

    Summary_Writer = AssistantAgent(
                name="Summary_Writer",
                model_client = model_qwen3_32B,
                tools=[],
                system_message="""
    # 角色定位
    作为总结者，你的任务是完整的整理新生成的对话语料。

    # 任务
    - 识别Client_Speaker和Counselor_Speaker的发言内容，也就是以"【来访者】："和"【咨询师】："开头的对话内容
    - 按照轮次，将他们的发言内容按顺序梳理成每一轮的对话形式。每一轮的对话一定是一次来访者发言和一次对应的咨询师发言。
    - 如实记录他们的发言内容，不要进行其他任何的总结概述或提炼
    - 严格按照顺序进行整理，确保每一轮对话中都是一次来访者发言和一次咨询师发言。不要丢失任何发言。

    # 输出要求
    输出格式清晰的多轮对话内容,标注好轮次和来访者与咨询师的发言""",
                description="""
    # 角色定位
    作为总结者，你的任务是完整的整理新生成的对话语料。
    """
            )


    Client_Profiler_f = MessageFilterAgent(
        name="Client_Profiler",
        wrapped_agent=Client_Profiler,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="user", position="first", count=1),
            ]
        ),
    )

    Client_Speaker_f = MessageFilterAgent(
        name="Client_Speaker",
        wrapped_agent=Client_Speaker,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="Client_Profiler", position="last", count=1),
                PerSourceFilter(source="Counselor_Speaker", position="last", count=1),
                PerSourceFilter(source="Client_Speaker", position="last", count=1),
                PerSourceFilter(source="Process_Monitor", position="last", count=1)
            ]
        ),
    )

    Safety_Monitor_f = MessageFilterAgent(
        name="Safety_Monitor",
        wrapped_agent=Safety_Monitor,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="Client_Speaker", position="last", count=1),
                PerSourceFilter(source="Process_Monitor", position="last", count=1),
            ]
        ),
    )

    Counselor_Supervisor_f = MessageFilterAgent(
        name="Counselor_Supervisor",
        wrapped_agent=Counselor_Supervisor,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="Client_Speaker", position="last", count=1),
                PerSourceFilter(source="Counselor_Speaker", position="last", count=1),
                PerSourceFilter(source="Safety_Monitor", position="last", count=1),
                PerSourceFilter(source="Process_Monitor", position="last", count=1),
                PerSourceFilter(source="Counselor_Supervisor", position="last", count=1),
            ]
        ),
    )

    Counselor_Speaker_f = MessageFilterAgent(
        name="Counselor_Speaker",
        wrapped_agent= Counselor_Speaker,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="Client_Speaker", position="last", count=1),
                PerSourceFilter(source="Counselor_Speaker", position="last", count=5),
                PerSourceFilter(source="Counselor_Supervisor", position="last", count=1),
                PerSourceFilter(source="Process_Monitor", position="last", count=1),
            ]
        ),
    )

    Process_Monitor_f = MessageFilterAgent(
        name="Process_Monitor",
        wrapped_agent= Process_Monitor,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="Client_Profiler", position="last", count=1),
                PerSourceFilter(source="Client_Speaker", position="last", count=15),
                PerSourceFilter(source="Counselor_Speaker", position="last", count=15),
                PerSourceFilter(source="Process_Monitor", position="last", count=2),
            ]
        ),
    )

    Summary_Writer_f = MessageFilterAgent(
        name="Summary_Writer",
        wrapped_agent= Summary_Writer,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="Client_Speaker", position="last", count=30),
                PerSourceFilter(source="Counselor_Speaker", position="last", count=30),
            ]
        ),
    )



    builder = DiGraphBuilder()
    builder.add_node(Client_Profiler_f).add_node(Process_Monitor_f).add_node(Client_Speaker_f).add_node(Safety_Monitor_f).add_node(Counselor_Supervisor_f).add_node(Counselor_Speaker_f).add_node(Summary_Writer_f)


    builder.add_edge(Client_Profiler_f, Process_Monitor_f)

    builder.add_edge(Process_Monitor_f, Client_Speaker_f, condition=lambda msg: "APPROVE" not in msg.to_model_text())

    builder.add_edge(Client_Speaker_f, Safety_Monitor_f)
    builder.add_edge(Safety_Monitor_f, Counselor_Supervisor_f)

    builder.add_edge(Counselor_Supervisor_f, Counselor_Speaker_f)

    builder.add_edge(Counselor_Speaker_f, Process_Monitor_f, activation_group="loop_back_group")

    builder.add_edge(Process_Monitor_f, Summary_Writer_f, condition=lambda msg: "APPROVE" in msg.to_model_text())

    builder.set_entry_point(Client_Profiler_f)

    graph = builder.build()

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=134) 
    termination = text_mention_termination | max_messages_termination

    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=termination
    )

    return flow, client_memory


async def main(ARCHETYPE_CN: str, ARCHETYPE_EN: str, item: dict, save_dir_path: str, knowledge_dir: str) -> None:

    conversations = item.get('conversations')
    assert conversations is not None
    task_str = """"""
    for i, conv in enumerate(conversations):
        client_speak = conv.get('Client')
        counselor_speak = conv.get('Counselor')
        conv_str = '第' + str(i+1) + '轮\n'
        conv_str += '来访者：' + client_speak + ' '
        conv_str += '咨询师：' + counselor_speak

        task_str += conv_str+'\n'

    item_id = item.get('id')
    
    # 记录原始stdout，防止串行执行时后续输出全进第一个日志
    original_stdout = sys.stdout
    log_file = redirect_output(ARCHETYPE_EN, save_dir_path, item_id)

    flow, client_memory = await create_psychological_flow(ARCHETYPE_CN, ARCHETYPE_EN, knowledge_dir)

    task = task_str

    agent_totals = {}
    agent_time_totals={}
    messages = []
    profiler_saved_to_memory = False

    time_0 = time.time()
    try:
        async for message in flow.run_stream(task=task):
            time_1 = time.time()
            # get the Client_Profiler's output, add to the memory of Client_Speaker
            if (not profiler_saved_to_memory) and hasattr(message, "source") and message.source == "Client_Profiler":
                if message.models_usage is not None:
                    try:
                        await client_memory.add(MemoryContent(
                            content=f"=== 来访者人物画像（由Client_Profiler生成） ===\n{message.content}\n=== 画像结束 ===",
                            mime_type=MemoryMimeType.TEXT
                        ))
                        profiler_saved_to_memory = True
                        print(f"[系统] 将人物画像写入Client_Speaker记忆成功！")
                    except Exception as e:
                        print(f"[系统] 将人物画像写入Client_Speaker记忆失败：{e}")


            if hasattr(message, 'source') and hasattr(message, 'content'):
                print(f"\n[{message.source}]: {message.content}")

                agent_time = message.source
                if agent_time not in agent_time_totals:
                    agent_time_totals[agent_time] = {
                        'time_cost': 0
                    }
                agent_time_totals[agent_time]['time_cost'] += (time_1 - time_0)

            messages.append(message)
            
            # collect token usage
            if hasattr(message, 'models_usage') and message.models_usage:
                usage = message.models_usage

                total = usage.prompt_tokens + usage.completion_tokens
                
                agent = message.source
                if agent not in agent_totals:
                    agent_totals[agent] = {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'call_count': 0,
                    }
                
                agent_totals[agent]['prompt_tokens'] += usage.prompt_tokens
                agent_totals[agent]['completion_tokens'] += usage.completion_tokens
                agent_totals[agent]['total_tokens'] += total
                agent_totals[agent]['call_count'] += 1

            time_0 = time.time()
        
        print("\n=== Dialogue Completed ===")
        print(f"Total message {len(messages)} processed.")

        print("\n=== Agent Time Cost ===")
        for agent, stats in agent_time_totals.items():
            print(f"[{agent}]:")
            print(f"   Time：{stats['time_cost']}")
        

        print("\n=== Agent Token Usage ===")
        total_all_tokens = 0
        for agent, stats in agent_totals.items():
            print(f"[{agent}]:")
            print(f"  Call count: {stats['call_count']}")
            print(f"  Total Token: prompt={stats['prompt_tokens']}, completion={stats['completion_tokens']}, total={stats['total_tokens']}")
            print(f"  Average: {stats['total_tokens']//stats['call_count'] if stats['call_count'] > 0 else 0} tokens")
            
            total_all_tokens += stats['total_tokens']
        
        print(f"\n=== Total ===")
        print(f"All Agent Total Token: {total_all_tokens}")
    except Exception as e:
        print(item_id, ' Process Error:', e)
    finally:
        # 恢复stdout，关闭文件
        sys.stdout = original_stdout
        log_file.close()

    flow = None
    client_memory = None


def run_one(item, save_dir, knowledge_dir, ARCHETYPE_EN2CN):

    disorder = item.get('disorder', '')
        
    disorder = disorder.strip('【】').split(',')[0]
    if disorder == '回避型人格':
        ARCHETYPE_EN = 'Avoidant'
    elif disorder == '反社会型人格':
        ARCHETYPE_EN = 'Antisocial'
    elif disorder == '边缘型人格':
        ARCHETYPE_EN = 'Borderline'
    elif disorder == '依赖型人格':
        ARCHETYPE_EN = 'Dependent'
    elif disorder == '表演型人格':
        ARCHETYPE_EN = 'Histrionic'
    elif disorder == '自恋型人格':
        ARCHETYPE_EN = 'Narcissistic'
    elif disorder == '强迫型人格':
        ARCHETYPE_EN = 'Obsessive-Compulsive'
    elif disorder == '偏执型人格':
        ARCHETYPE_EN = 'Paranoid'
    elif disorder == '类分裂样人格':
        ARCHETYPE_EN = 'Schizoid'
    elif disorder == '分裂型人格':
        ARCHETYPE_EN = 'Schizotypal'
    else:
        return {"id": item.get("id"), "status": "skipped", "reason": f"unknown disorder: {disorder}"}

    try:
        asyncio.run(main(
            ARCHETYPE_CN=ARCHETYPE_EN2CN[ARCHETYPE_EN],
            ARCHETYPE_EN=ARCHETYPE_EN,
            item=item,
            save_dir_path=save_dir,
            knowledge_dir=knowledge_dir
        ))
        return {"id": item.get("id"), "status": "ok"}
    except Exception as e:
        return {"id": item.get("id"), "status": "error", "error": str(e)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate psychological dialogues')
    parser.add_argument('--seed_path',
                       type=str,
                       default='./DataSeed.json',
                       help='path to the seed data file')
    parser.add_argument('--save_dir',
                       type=str,
                       default='./Results/',
                       help='dir to save the generated logs')
    parser.add_argument('--knowledge_dir',
                       type=str,
                       default='./psy_knowledge/',
                       help='dir containing the psychological knowledge txt files')
    parser.add_argument('--max_case',
                       type=int,
                       default=1)
    args, unparsed = parser.parse_known_args()

    ARCHETYPE_EN2CN = {
        "Dependent" : "依赖型",
        "Paranoid" : "偏执型",
        "Antisocial" : "反社会型",
        "Avoidant" : "回避型",
        "Histrionic" : "表演型",
        "Borderline" : "边缘型",
        "Obsessive-Compulsive" : "强迫型",
        "Narcissistic" : "自恋型",
        "Schizotypal" : "分裂型",
        "Schizoid" : "类分裂样"
    }

    print(f"Reading seed data from {args.seed_path}...")
    with open(args.seed_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # limit the max_case in a running time (if necessary)
    # if args.max_case is not None and args.max_case > 0:
    #     data = data[:args.max_case]

    results = []
    
    for item in tqdm(data, desc="Processing..."):
        result = run_one(item, args.save_dir, args.knowledge_dir, ARCHETYPE_EN2CN)
        results.append(result)
        if result.get('status') == 'error':
            print(f"\\n[Error] Failed on item {item.get('id')}: {result.get('error')}")

    print("\\nAll Done! Results saved to:", args.save_dir)
