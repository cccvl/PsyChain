# PsyChain: A Collaborative Chain-of-Agents Framework for Generating Personalized and Professional Counseling Dialogues

<p align="center">
  <img src="framework_PsyChain.jpg" alt="PsyChain Framework" width="80%">
</p>

> 🎉 **Accepted at ACL 2026 Findings**

## Overview
Existing psychological counseling datasets often suffer from monolithic client personas, insufficient therapeutic depth, and a lack of process controllability. To address these critical limitations, we propose **PsyChain**, a chain-of-agents framework that evolves static counseling corpora into high-fidelity dialogues through collaborative simulation which explicitly models client personality, stage progression, safety monitoring, and expert supervision. PsyChain involves a Client Profiler that extracts life scenarios and pairs them with psychological personality archetypes to synthesize diverse profiles. To simulate the complete counseling process, five specialized agents—Process Monitor, Client Speaker, Safety Monitor, Counselor Supervisor, and Counselor Speaker—collaborate and interact autonomously at each dialogue turn to ensure therapeutic professionalism and safety. We apply this to construct **PsyChainD**, a Chinese dataset of 10,456 dialogues featuring systematically diverse client profiles. Extensive evaluation across *client side*, *counselor side* and *overall quality* shows substantial improvements. The model trained on PsyChainD achieves 61-91% win rates against domain-specific baselines in pairwise evaluation and the highest average score in human evaluation, indicating potential for real-world counseling.

## Environment Requirements
- Python version 3.11
- AutoGen version 0.7.4
- Other dependencies are listed in `requirements.txt`

## Repository Status
✅ **Available now:**
- Dialogue generation code implemented with AutoGen (`generate_dialogues.py`)
- Dialogue quality evaluation code (`evaluation.py`)

⏳ **Coming soon:**
- Generated counseling dataset PsyChainD (Test split)
- Detailed usage instruction

## Contact
For questions or issues, please open an issue in this repository
