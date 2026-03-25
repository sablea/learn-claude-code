#!/usr/bin/env python3
"""
industrial_retrieval_agent.py - 工业场景数据检索 Agent (~500 lines)

项目背景
=======
工业场景下，技术员需要快速查询设备信息、历史维护记录、故障报告和
标准作业程序（SOP）。数据分散在两类存储中：

  1. SQL 数据库 - 结构化数据：设备台账、维护记录、故障代码、零件库存
  2. 文档库      - 非结构化数据：故障报告、SOP、设备手册

本 Agent 基于 learn-claude-code 教程的 Agent 模式（v1/v4）实现，
提供统一的自然语言查询接口，内部自动路由到合适的检索工具。

架构
----
  用户自然语言查询
       |
       v
  Agent Loop (LLM 决策)
       |
   ____+____
  |         |
  v         v
sql_query  rag_search   ← 两大检索工具
  |         |
  v         v
SQLite    TF-IDF 文档索引

工具列表
--------
  | 工具            | 用途                           |
  |-----------------|-------------------------------|
  | sql_query       | 执行 SQL 查询，返回结构化结果   |
  | rag_search      | 语义搜索文档（TF-IDF 相似度）  |
  | list_tables     | 列出数据库表及结构              |
  | list_documents  | 列出可检索的文档列表            |
  | Skill           | 加载工业检索领域知识            |

Usage:
    python industrial_retrieval_agent.py
"""

import math
import os
import re
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================================================
# 配置
# =============================================================================

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
DATA_DIR = WORKDIR / "data"
DOCS_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "industrial.db"

MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


# =============================================================================
# 示例数据 - 内置工业场景样本数据
# =============================================================================

SAMPLE_EQUIPMENT = [
    ("CNC001", "数控铣床 #1", "VMC-850", "A车间", "operational", "2020-03-15", "2024-11-20"),
    ("CNC002", "数控铣床 #2", "VMC-850", "A车间", "fault", "2020-03-15", "2024-09-10"),
    ("HYD001", "液压压力机 #1", "HYP-200T", "B车间", "maintenance", "2019-07-01", "2024-12-01"),
    ("HYD002", "液压压力机 #2", "HYP-200T", "B车间", "operational", "2019-07-01", "2024-10-15"),
    ("ROB001", "焊接机器人 #1", "ABB-IRB6700", "C车间", "operational", "2021-05-20", "2024-11-05"),
    ("ROB002", "焊接机器人 #2", "ABB-IRB6700", "C车间", "offline", "2021-05-20", "2024-08-30"),
    ("CONV001", "传送带系统", "CTB-500", "D车间", "operational", "2018-11-10", "2024-12-10"),
    ("COMP001", "空压机 #1", "Atlas-GA45", "机房", "operational", "2017-06-01", "2024-11-28"),
]

SAMPLE_MAINTENANCE = [
    ("CNC001", "2024-11-20", "preventive", "定期润滑主轴、清洁冷却系统、校准刀具补偿", "张工", 1200.0, 4.0),
    ("CNC001", "2024-08-15", "corrective", "更换主轴轴承，消除异常振动", "李工", 5800.0, 8.0),
    ("CNC001", "2024-05-10", "preventive", "更换液压油、检查各轴精度", "张工", 800.0, 3.0),
    ("CNC002", "2024-09-10", "preventive", "清洁过滤器、润滑导轨", "张工", 600.0, 2.0),
    ("CNC002", "2024-12-15", "emergency", "主轴驱动故障E003，更换驱动模块", "王工", 12000.0, 12.0),
    ("HYD001", "2024-12-01", "preventive", "更换液压油、检查密封件、测试安全阀", "陈工", 3500.0, 6.0),
    ("HYD001", "2024-07-20", "corrective", "修复液压泵泄漏，更换密封圈", "陈工", 2200.0, 5.0),
    ("ROB001", "2024-11-05", "preventive", "润滑各轴关节、检查焊枪冷却系统", "刘工", 1500.0, 4.0),
    ("ROB001", "2024-06-18", "corrective", "更换焊枪导电嘴、清理焊渣堆积", "刘工", 800.0, 3.0),
    ("ROB002", "2024-08-30", "corrective", "控制系统故障，等待备件，暂停使用", "刘工", 0.0, 2.0),
    ("COMP001", "2024-11-28", "preventive", "更换空气过滤器、油过滤器，检查安全阀", "赵工", 1800.0, 3.0),
]

SAMPLE_FAULT_CODES = [
    ("E001", "CNC", "主轴过热报警", "critical", "立即停机，检查冷却系统是否堵塞，清洁散热器"),
    ("E002", "CNC", "刀具破损检测", "warning", "暂停加工，检查刀具状态，必要时更换刀具"),
    ("E003", "CNC", "主轴驱动故障", "critical", "停机断电，联系维修人员，检查驱动模块和连接线缆"),
    ("E004", "CNC", "X轴超程报警", "warning", "回零操作，检查行程开关和限位设置"),
    ("E005", "CNC", "液压压力不足", "warning", "检查液压油位，确认液压泵工作状态"),
    ("H001", "HYD", "液压系统过压", "critical", "立即卸压停机，检查溢流阀设定值，排查管路堵塞"),
    ("H002", "HYD", "液压油温过高", "warning", "检查冷却器是否正常，减少工作频率"),
    ("H003", "HYD", "液压泵异常噪声", "warning", "检查油位，排查气穴现象，必要时更换液压泵"),
    ("R001", "ROB", "关节力矩超限", "critical", "停止运动，检查夹具是否碰撞，检查路径规划"),
    ("R002", "ROB", "焊接参数偏差", "warning", "检查焊丝送丝系统，校准焊接参数"),
    ("C001", "COMP", "排气温度过高", "critical", "停机冷却，检查冷却风扇和散热器"),
    ("C002", "COMP", "油分离器堵塞", "warning", "更换油分离器滤芯"),
]

SAMPLE_PARTS = [
    ("BRG-001", "主轴轴承 6205-2RS", 8, "VMC-850", 3),
    ("BRG-002", "导轨滚珠 Φ6.35", 500, "VMC-850", 100),
    ("SEL-001", "液压泵密封圈套装", 5, "HYP-200T", 2),
    ("FLT-001", "液压油过滤器 HF7920", 12, "HYP-200T,Atlas-GA45", 4),
    ("FLT-002", "空气过滤器 1614905600", 6, "Atlas-GA45", 3),
    ("OIL-001", "液压油 HM46 (20L)", 10, "HYP-200T", 4),
    ("OIL-002", "主轴润滑油 ISO VG32 (5L)", 15, "VMC-850", 5),
    ("DRV-001", "主轴变频驱动模块", 1, "VMC-850", 1),
    ("WLD-001", "焊枪导电嘴 M6×25", 200, "ABB-IRB6700", 50),
    ("CTB-001", "传送带驱动皮带 B型", 4, "CTB-500", 2),
]

SAMPLE_DOCUMENTS = {
    "fault_report_2024_001.txt": """故障报告
==========
报告编号：FR-2024-001
日期：2024-08-15
设备：CNC001（数控铣床 #1，VMC-850）
报告人：李工
审核：车间主任

一、故障现象
设备在加工过程中突然报E001（主轴过热）报警，主轴自动停止，
加工中心停机。操作人员反映加工前已连续运行约4小时。

二、故障原因分析
检查发现：
1. 主轴冷却液喷嘴部分堵塞，冷却效果下降约40%
2. 主轴内置风扇散热鳍片积累大量铝屑，影响散热
3. 近期加工铝合金件数量增加，产生切屑较多

三、处理措施
1. 清洁冷却液喷嘴，疏通堵塞
2. 拆解清洁主轴风扇散热鳍片
3. 更换主轴轴承（因检查发现轴承游隙已超标）
4. 检查并补充主轴润滑油

四、修复结果
经上述处理后，主轴温升恢复正常范围（<45°C），设备恢复生产。

五、预防建议
1. 每日班前检查冷却液喷嘴状态
2. 每周清洁主轴散热鳍片
3. 加工铝合金时适当降低切削参数，减少热量产生
""",

    "fault_report_2024_002.txt": """故障报告
==========
报告编号：FR-2024-002
日期：2024-12-15
设备：CNC002（数控铣床 #2，VMC-850）
报告人：王工
审核：车间主任

一、故障现象
设备突然报E003（主轴驱动故障）报警，控制系统显示主轴驱动模块
通信中断，设备无法启动。操作人员反映故障发生前设备有短暂抖动。

二、故障原因分析
经检测：
1. 主轴变频驱动模块（DRV-001）内部IGBT模块击穿
2. 分析原因：驱动柜散热风扇故障，长期过热导致器件损坏
3. 驱动模块已使用4年余，接近设计寿命

三、处理措施
1. 更换主轴变频驱动模块（DRV-001）
2. 同时更换散热风扇
3. 清洁驱动柜内部积尘

四、修复结果
更换驱动模块后，主轴恢复正常运转，各参数正常。

五、预防建议
1. 每季度检查驱动柜散热风扇状态
2. 建立关键电气元件寿命台账，提前预防性更换
3. 保持驱动柜周围通风良好，环境温度不超过40°C
""",

    "fault_report_2024_003.txt": """故障报告
==========
报告编号：FR-2024-003
日期：2024-07-20
设备：HYD001（液压压力机 #1，HYP-200T）
报告人：陈工
审核：车间主任

一、故障现象
设备在生产过程中发现液压系统底部有油迹，操作人员检查发现液压泵
连接处有明显渗漏，系统压力略有下降。

二、故障原因分析
1. 液压泵进出油口密封圈老化失效
2. 设备安装至今已超过5年，密封件老化属正常现象
3. 近期高温天气加速了密封材料老化

三、处理措施
1. 停机泄压，拆卸液压泵
2. 更换液压泵全套密封圈（SEL-001）
3. 更换液压油（OIL-001），检查油路清洁度
4. 重新调整系统压力至设定值200T

四、修复结果
更换密封件后无渗漏，系统压力稳定，恢复正常生产。

五、预防建议
1. 每季度检查液压系统所有连接部位
2. 根据厂家建议，每2年更换一次液压泵密封件
3. 保持液压油清洁，定期检测油液污染度
""",

    "sop_preventive_maintenance_cnc.txt": """标准作业程序（SOP）
=====================
文档编号：SOP-MNT-001
版本：V2.1
适用设备：数控铣床（VMC-850系列）
生效日期：2024-01-01

一、目的
规范数控铣床预防性维护操作，确保设备长期稳定运行，预防突发故障。

二、适用范围
本SOP适用于A车间所有VMC-850系列数控铣床（CNC001、CNC002）。

三、维护周期与内容

【日常检查（每班次）】
1. 检查冷却液液位，不足时补充
2. 检查主轴润滑油压力表读数（正常范围：0.1-0.3MPa）
3. 清除机床导轨上的切屑和冷却液
4. 检查气压系统压力（正常范围：0.5-0.7MPa）
5. 观察各轴运动是否顺畅，有无异常噪音

【周检（每周一）】
1. 清洁主轴散热鳍片和冷却风扇
2. 检查并清洁冷却液喷嘴，防止堵塞
3. 检查导轨防尘刮板完好性
4. 润滑丝杠螺母（按设备手册规定用量）
5. 检查刀库运动是否顺畅

【月检（每月第一个星期一）】
1. 更换或清洁主轴油雾分离器
2. 检查液压系统油位和压力
3. 校验工件坐标系（G54等）
4. 检查主轴径向跳动（标准：≤0.002mm）
5. 润滑主轴轴承

【季检（每季度）】
1. 检查驱动柜散热风扇
2. 清洁驱动柜内部积尘
3. 检查全部伺服电机温升（正常范围：<65°C）
4. 检查并紧固所有电气连接端子
5. 备份机床参数（系统参数、刀补数据）

四、注意事项
1. 所有维护操作前必须确认设备处于停机状态，按急停按钮
2. 清洁时禁止使用压缩空气直接吹向电气元件
3. 发现异常立即停止维护，上报车间主任
4. 维护后填写设备维护记录表

五、所需工具和材料
- 主轴润滑油（ISO VG32）
- 导轨润滑脂（锂基脂2号）
- 清洁剂（不含氯）
- 清洁布（无纺布）
- 量块（用于检验精度）
""",

    "sop_emergency_shutdown.txt": """标准作业程序（SOP）
=====================
文档编号：SOP-EMG-001
版本：V1.3
适用设备：所有生产设备
生效日期：2024-01-01

一、目的
规范紧急停机操作程序，确保人员安全和设备安全，减少突发事故损失。

二、紧急停机触发条件

以下任何情况必须立即执行紧急停机：
1. 人身安全受到威胁
2. 设备发出异常噪音或振动
3. 发生漏电、冒烟、起火
4. 液压系统爆管或大量泄漏
5. 控制系统报critical级别故障码
6. 操作人员无法控制设备运动

三、通用紧急停机步骤

第一步：按下就近急停按钮（红色蘑菇头按钮）
  - 急停按钮位于操作面板正面
  - 按下后顺时针旋转锁定
  - 确认设备所有运动已停止

第二步：切断设备主电源
  - 在配电柜将设备主开关扳至"OFF"位置
  - 等待控制系统完全断电（约10秒）

第三步：处理液压/气压系统
  - 液压设备：打开卸压阀，使系统压力降至零
  - 气压设备：关闭气源阀门，排尽残余气压

第四步：上报和记录
  - 立即通知车间主任和维修人员
  - 记录停机时间、现象和操作过程
  - 未经维修人员确认，禁止重新启动

四、液压压力机紧急停机（HYP-200T专项）

1. 确认压机滑块处于上死点或安全位置
2. 按下急停按钮
3. 将液压系统选择开关切至"手动卸压"
4. 缓慢打开卸压阀，确认压力表归零
5. 关闭液压泵电机
6. 将蓄能器压力完全释放

五、数控铣床紧急停机（VMC-850专项）

1. 按下急停按钮，主轴和各轴立即停止
2. 检查刀具是否仍在工件中，如有则手动退刀
3. 打开防护门（确认主轴已停止）
4. 切断伺服电源开关
5. 保留故障报警界面截图，便于诊断

六、注意事项
1. 急停后禁止在未排查故障前复位急停按钮
2. 液压系统必须完全泄压后才能进行检修
3. 紧急停机后必须填写《设备异常报告单》
""",

    "sop_hydraulic_maintenance.txt": """标准作业程序（SOP）
=====================
文档编号：SOP-MNT-002
版本：V1.5
适用设备：液压压力机（HYP-200T系列）
生效日期：2024-01-01

一、目的
规范液压压力机维护保养操作，延长设备使用寿命，保证生产安全。

二、液压系统日常检查

【每班次】
1. 检查液压油箱油位（油位计应在上下限标记之间）
2. 检查系统工作压力（正常范围：180-200bar）
3. 检查油温（正常范围：35-55°C）
4. 观察液压管路有无渗漏
5. 监听液压泵运行声音，有无异常噪音

【每周】
1. 清洁油箱通气过滤器
2. 检查蓄能器预充压力（标准：80bar氮气）
3. 检查安全阀设定值
4. 清洁机身表面油污

【每季度】
1. 更换液压油过滤器（FLT-001）
2. 检测液压油污染度（目标：NAS 7级以内）
3. 检查所有密封件状态
4. 测试各安全联锁功能

【每年】
1. 更换液压油（OIL-001）
2. 清洁油箱内部
3. 更换液压泵密封件（SEL-001）
4. 全面检查液压阀组

三、液压油更换步骤

1. 设备停机，系统完全泄压
2. 排放旧液压油（使用油桶接收，不得污染地面）
3. 清洁油箱内壁
4. 更换液压油过滤器
5. 注入新液压油至上限位置
6. 启动液压泵，低压循环10分钟
7. 检查系统是否有渗漏
8. 调整系统压力至正常值
9. 填写保养记录

四、常见故障排查

| 故障现象 | 可能原因 | 检查方法 |
|----------|----------|----------|
| 压力不足 | 泵磨损/溢流阀失调 | 测量泵输出压力 |
| 油温过高 | 冷却器脏/油位低 | 清洁冷却器，检查油位 |
| 噪音大   | 气穴/轴承磨损 | 检查油位，监测振动 |
| 爬行     | 系统有空气 | 排气操作 |
| 泄漏     | 密封件老化 | 检查所有密封部位 |
""",

    "equipment_manual_cnc_vmc850.txt": """设备技术手册（节选）
===================
设备型号：VMC-850
制造商：XXX数控机床有限公司
文档版本：Rev.3.0

一、技术规格

主轴参数：
- 主轴转速范围：50-12000 RPM
- 主轴功率：15kW（额定），18.5kW（最大）
- 主轴锥孔：BT40
- 主轴轴承：陶瓷轴承，预压等级P2

行程参数：
- X轴行程：850mm
- Y轴行程：500mm
- Z轴行程：500mm
- 工作台尺寸：950×500mm

精度参数：
- 定位精度：±0.008mm
- 重复定位精度：±0.004mm
- 主轴径向跳动：≤0.002mm

二、润滑系统说明

主轴润滑（主轴油雾润滑）：
- 使用油品：ISO VG32主轴油
- 油压：0.1-0.3MPa
- 油温报警：>70°C

导轨润滑（集中润滑）：
- 使用油品：导轨油68#
- 润滑周期：每30分钟自动润滑一次
- 单次供油量：0.5ml/轴

三、日常保养要点

冷却系统：
1. 每日检查冷却液浓度（推荐5-8%）
2. 冷却液使用寿命约6个月，定期更换
3. 每周清洁冷却箱，防止切屑积累
4. 冷却液喷嘴每日检查是否堵塞

主轴维护：
1. 主轴径向跳动超过0.005mm时需更换轴承
2. 主轴轴承寿命约20000小时
3. 更换轴承必须在专业维修人员监督下进行

四、故障代码详解

E001 - 主轴过热（>80°C）
  原因：冷却不足、轴承故障、切削参数过大
  处理：停机检查冷却系统，必要时更换轴承

E002 - 刀具破损
  原因：刀具磨损过度、切削参数不当
  处理：检查刀具，调整切削参数

E003 - 主轴驱动故障
  原因：驱动模块故障、编码器故障
  处理：联系专业维修人员，不得自行拆卸驱动模块

E004 - 行程超限
  原因：程序错误、限位开关故障
  处理：手动回零，检查限位开关

五、主轴轴承更换程序（专业人员操作）

1. 停机，断开主轴驱动电源
2. 拆除主轴电机连接
3. 拆卸主轴单元（需专用工具）
4. 在恒温室（20°C±1°C）更换轴承
5. 重新装配，注意预压调整
6. 动平衡测试（G2.5级）
7. 安装后试运行，监测温升和振动
""",
}


# =============================================================================
# 数据库初始化
# =============================================================================

def init_database(db_path: Path) -> None:
    """
    初始化工业设备数据库，创建表并插入示例数据。

    如果数据库已存在，则跳过初始化（幂等操作）。
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 创建表
    c.executescript("""
        CREATE TABLE IF NOT EXISTS equipment (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            model TEXT,
            location TEXT,
            status TEXT,
            install_date TEXT,
            last_maintenance_date TEXT
        );

        CREATE TABLE IF NOT EXISTS maintenance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_id TEXT,
            date TEXT,
            type TEXT,
            description TEXT,
            technician TEXT,
            cost REAL,
            duration_hours REAL
        );

        CREATE TABLE IF NOT EXISTS fault_codes (
            code TEXT PRIMARY KEY,
            equipment_type TEXT,
            description TEXT,
            severity TEXT,
            recommended_action TEXT
        );

        CREATE TABLE IF NOT EXISTS parts_inventory (
            part_id TEXT PRIMARY KEY,
            name TEXT,
            quantity INTEGER,
            compatible_equipment TEXT,
            reorder_point INTEGER
        );
    """)

    # 仅在表为空时插入示例数据
    if not c.execute("SELECT 1 FROM equipment LIMIT 1").fetchone():
        c.executemany(
            "INSERT INTO equipment VALUES (?,?,?,?,?,?,?)",
            SAMPLE_EQUIPMENT
        )
    if not c.execute("SELECT 1 FROM maintenance_records LIMIT 1").fetchone():
        c.executemany(
            "INSERT INTO maintenance_records "
            "(equipment_id,date,type,description,technician,cost,duration_hours) "
            "VALUES (?,?,?,?,?,?,?)",
            SAMPLE_MAINTENANCE
        )
    if not c.execute("SELECT 1 FROM fault_codes LIMIT 1").fetchone():
        c.executemany(
            "INSERT INTO fault_codes VALUES (?,?,?,?,?)",
            SAMPLE_FAULT_CODES
        )
    if not c.execute("SELECT 1 FROM parts_inventory LIMIT 1").fetchone():
        c.executemany(
            "INSERT INTO parts_inventory VALUES (?,?,?,?,?)",
            SAMPLE_PARTS
        )

    conn.commit()
    conn.close()


def init_documents(docs_dir: Path) -> None:
    """
    初始化文档库，将示例文档写入磁盘（幂等操作）。
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in SAMPLE_DOCUMENTS.items():
        doc_path = docs_dir / filename
        if not doc_path.exists():
            doc_path.write_text(content, encoding="utf-8")


# =============================================================================
# TF-IDF 文档检索引擎（无额外依赖）
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """
    对中英文混合文本进行分词。
    中文按字符切分，英文和数字保留完整词。
    """
    tokens = []
    # 提取英文单词和数字
    for word in re.findall(r'[a-zA-Z0-9_\-]+', text.lower()):
        tokens.append(word)
    # 提取中文字符（也可按词组，但这里简化为按字）
    for char in re.findall(r'[\u4e00-\u9fff]', text):
        tokens.append(char)
    return tokens


class RAGIndex:
    """
    基于 TF-IDF 的轻量级文档检索引擎。

    实现：
      1. 加载文档 -> 分词 -> 建立词频索引
      2. 计算 IDF（逆文档频率）
      3. 查询时计算 TF-IDF 余弦相似度，返回 Top-K 结果
    """

    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.documents: dict[str, str] = {}    # filename -> content
        self.doc_tokens: dict[str, list] = {}   # filename -> tokens
        self.idf: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        """加载所有文档并构建 TF-IDF 索引。"""
        if not self.docs_dir.exists():
            return

        for path in sorted(self.docs_dir.glob("*.txt")):
            content = path.read_text(encoding="utf-8")
            self.documents[path.name] = content
            self.doc_tokens[path.name] = _tokenize(content)

        # 计算 IDF
        n = len(self.doc_tokens)
        if n == 0:
            return
        all_terms = set(t for tokens in self.doc_tokens.values() for t in tokens)
        for term in all_terms:
            df = sum(1 for tokens in self.doc_tokens.values() if term in tokens)
            self.idf[term] = math.log((n + 1) / (df + 1)) + 1

    def reload(self) -> None:
        """重新加载文档索引（文档有变更时调用）。"""
        self.documents.clear()
        self.doc_tokens.clear()
        self.idf.clear()
        self._load()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        TF-IDF 检索，返回最相关的 top_k 个文档片段。

        返回：[{filename, score, snippet}, ...]
        """
        if not self.documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for fname, tokens in self.doc_tokens.items():
            tf = Counter(tokens)
            total = max(len(tokens), 1)
            score = sum(
                (tf.get(t, 0) / total) * self.idf.get(t, 0)
                for t in query_tokens
            )
            scores.append((fname, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for fname, score in scores[:top_k]:
            if score <= 0:
                continue
            content = self.documents[fname]
            # 提取最相关片段（包含查询词的上下文）
            snippet = _extract_snippet(content, query_tokens, max_chars=500)
            results.append({
                "filename": fname,
                "score": round(score, 4),
                "snippet": snippet,
                "full_content": content,
            })
        return results

    def list_documents(self) -> list[dict]:
        """列出所有已索引文档。"""
        result = []
        for fname in sorted(self.documents.keys()):
            lines = self.documents[fname].splitlines()
            preview = " ".join(lines[:3])[:100]
            result.append({
                "filename": fname,
                "size_chars": len(self.documents[fname]),
                "preview": preview,
            })
        return result


def _extract_snippet(content: str, query_tokens: list, max_chars: int = 500) -> str:
    """提取内容中包含查询词的最相关段落片段。"""
    lines = content.splitlines()
    query_set = set(query_tokens)
    best_line_idx = 0
    best_overlap = -1

    for i, line in enumerate(lines):
        overlap = len(query_set & set(_tokenize(line)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_line_idx = i

    # 取最相关行周围的上下文
    start = max(0, best_line_idx - 2)
    end = min(len(lines), best_line_idx + 8)
    snippet = "\n".join(lines[start:end])
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return snippet


# =============================================================================
# Skills 加载器（复用 v4 的 SkillLoader）
# =============================================================================


class SkillLoader:
    """从 SKILL.md 文件加载领域知识（与 v4 相同的机制）。"""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict = {}
        self._load()

    def _parse(self, path: Path) -> dict | None:
        content = path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None
        frontmatter, body = match.groups()
        meta = {}
        for line in frontmatter.strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip("\"'")
        if "name" not in meta or "description" not in meta:
            return None
        return {"name": meta["name"], "description": meta["description"], "body": body.strip()}

    def _load(self) -> None:
        if not self.skills_dir.exists():
            return
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                skill = self._parse(skill_md)
                if skill:
                    self.skills[skill["name"]] = skill

    def get_descriptions(self) -> str:
        if not self.skills:
            return "(暂无可用技能)"
        return "\n".join(
            f"- {name}: {s['description']}"
            for name, s in self.skills.items()
        )

    def get_skill_content(self, name: str) -> str | None:
        if name not in self.skills:
            return None
        s = self.skills[name]
        return f"# Skill: {s['name']}\n\n{s['body']}"


# =============================================================================
# 全局单例
# =============================================================================

# 初始化数据库和文档库
init_database(DB_PATH)
init_documents(DOCS_DIR)

SKILLS = SkillLoader(SKILLS_DIR)
RAG = RAGIndex(DOCS_DIR)


# =============================================================================
# 系统提示词
# =============================================================================

SYSTEM = f"""你是工业设备管理系统的智能检索助手。

你的职责：帮助技术员快速查询设备信息、故障记录、维护历史和操作规程。

**可用技能**（调用 Skill 工具加载）：
{SKILLS.get_descriptions()}

**检索工具**：
- sql_query: 查询结构化数据（设备台账、维护记录、故障代码、零件库存）
- rag_search: 搜索文档（故障报告、SOP、设备手册）
- list_tables: 查看数据库表结构
- list_documents: 查看可检索的文档列表

**工作准则**：
1. 先用 Skill 工具加载 industrial-retrieval 技能，获取数据库结构和查询指导
2. 根据问题类型选择合适工具：结构化数据用 sql_query，文档用 rag_search
3. 复杂问题同时使用 SQL 和 RAG，综合多源信息回答
4. 用清晰的中文回答，关键数据用表格展示
5. 如果找不到信息，明确告知用户"""


# =============================================================================
# 工具定义
# =============================================================================

TOOLS = [
    {
        "name": "sql_query",
        "description": "执行 SQL 查询，检索设备台账、维护记录、故障代码或零件库存。"
                       "使用参数化查询时，将参数放在 params 列表中。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL 查询语句（只允许 SELECT，禁止修改数据）"
                },
                "params": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "SQL 参数列表（用于参数化查询，如 WHERE id = ?）",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rag_search",
        "description": "在文档库中搜索相关内容。适用于查找故障报告、操作规程（SOP）、"
                       "设备手册等非结构化文档。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言搜索查询，描述你想找的内容"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回最相关的文档数量（默认3，最大5）",
                },
                "full_content": {
                    "type": "boolean",
                    "description": "是否返回完整文档内容（默认 false，返回摘要片段）",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_tables",
        "description": "列出数据库中所有表的名称和字段结构，帮助了解可查询的数据范围。",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "（可选）指定表名，查看详细结构和示例数据",
                },
            },
        },
    },
    {
        "name": "list_documents",
        "description": "列出文档库中所有可检索的文档，显示文件名和内容预览。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "Skill",
        "description": f"加载领域知识技能，获取专业指导。\n\n可用技能：\n{SKILLS.get_descriptions()}",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "技能名称（如 industrial-retrieval）"
                },
            },
            "required": ["skill"],
        },
    },
]


# =============================================================================
# 工具实现
# =============================================================================

def run_sql_query(query: str, params: list | None = None) -> str:
    """
    执行只读 SQL 查询。

    安全措施：仅允许 SELECT 语句，禁止 DDL/DML 操作。
    结果格式化为人类可读的表格。
    """
    query_stripped = query.strip().upper()
    if not query_stripped.startswith("SELECT"):
        return "错误：只允许执行 SELECT 查询，不允许修改数据。"

    # 阻止危险关键词（额外安全防护）
    dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "ATTACH"]
    if any(kw in query_stripped for kw in dangerous):
        return "错误：查询包含不允许的操作关键词。"

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute(query, params or [])
        rows = c.fetchall()
        conn.close()

        if not rows:
            return "查询结果为空。"

        # 格式化为文本表格
        columns = rows[0].keys()
        col_widths = {col: len(col) for col in columns}
        for row in rows:
            for col in columns:
                col_widths[col] = max(col_widths[col], len(str(row[col] or "")))

        lines = []
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        separator = "-+-".join("-" * col_widths[col] for col in columns)
        lines.append(header)
        lines.append(separator)
        for row in rows:
            line = " | ".join(str(row[col] or "").ljust(col_widths[col]) for col in columns)
            lines.append(line)

        lines.append(f"\n共 {len(rows)} 条记录")
        return "\n".join(lines)

    except sqlite3.Error as e:
        return f"SQL 执行错误：{e}"
    except Exception as e:
        return f"错误：{e}"


def run_rag_search(query: str, top_k: int = 3, full_content: bool = False) -> str:
    """
    TF-IDF 文档检索，返回最相关文档片段。
    """
    top_k = min(max(1, top_k), 5)
    results = RAG.search(query, top_k=top_k)

    if not results:
        return f"未找到与 '{query}' 相关的文档。"

    lines = [f"搜索 '{query}' 找到 {len(results)} 个相关文档：\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{'='*60}")
        lines.append(f"[{i}] 文档：{r['filename']}  (相关度：{r['score']})")
        lines.append("-" * 60)
        if full_content:
            lines.append(r["full_content"])
        else:
            lines.append(r["snippet"])
    lines.append("=" * 60)
    return "\n".join(lines)


def run_list_tables(table_name: str | None = None) -> str:
    """
    列出数据库表结构，可选显示指定表的详细信息和示例数据。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        if table_name:
            # 查看指定表的结构
            c.execute(f"PRAGMA table_info({table_name})")
            cols = c.fetchall()
            if not cols:
                return f"表 '{table_name}' 不存在。"

            lines = [f"表：{table_name}", "-" * 40]
            lines.append(f"{'列名':<25} {'类型':<10} {'非空':<6} {'默认值'}")
            lines.append("-" * 60)
            for col in cols:
                lines.append(
                    f"{col['name']:<25} {col['type']:<10} "
                    f"{'YES' if col['notnull'] else 'NO':<6} {col['dflt_value'] or ''}"
                )

            # 显示前3条示例数据
            c.execute(f"SELECT * FROM {table_name} LIMIT 3")
            samples = c.fetchall()
            if samples:
                lines.append(f"\n示例数据（前 {len(samples)} 条）：")
                keys = samples[0].keys()
                lines.append(" | ".join(str(k)[:15] for k in keys))
                for row in samples:
                    lines.append(" | ".join(str(row[k] or "")[:15] for k in keys))

        else:
            # 列出所有表
            c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in c.fetchall()]

            lines = ["数据库中的表：", "=" * 40]
            for tbl in tables:
                c.execute(f"SELECT COUNT(*) FROM {tbl}")
                count = c.fetchone()[0]
                c.execute(f"PRAGMA table_info({tbl})")
                col_names = [col[1] for col in c.fetchall()]
                lines.append(f"\n📋 {tbl}（{count} 条记录）")
                lines.append(f"   字段：{', '.join(col_names)}")

        conn.close()
        return "\n".join(lines)

    except Exception as e:
        return f"错误：{e}"


def run_list_documents() -> str:
    """列出文档库中的所有文档。"""
    docs = RAG.list_documents()
    if not docs:
        return "文档库为空。"

    lines = [f"文档库共 {len(docs)} 个文档：", "=" * 60]
    for doc in docs:
        lines.append(f"\n📄 {doc['filename']}（{doc['size_chars']} 字符）")
        lines.append(f"   {doc['preview']}...")
    return "\n".join(lines)


def run_skill(skill_name: str) -> str:
    """加载技能知识，注入到对话上下文（缓存友好方式）。"""
    content = SKILLS.get_skill_content(skill_name)
    if not content:
        available = ", ".join(SKILLS.skills.keys()) or "（无可用技能）"
        return f"技能 '{skill_name}' 不存在。可用技能：{available}"
    return (
        f'<skill-loaded name="{skill_name}">\n'
        f"{content}\n"
        f"</skill-loaded>\n\n"
        f"已加载技能 '{skill_name}'，请按照技能指导进行操作。"
    )


def execute_tool(name: str, args: dict) -> str:
    """分发工具调用到对应实现。"""
    if name == "sql_query":
        return run_sql_query(args["query"], args.get("params"))
    if name == "rag_search":
        return run_rag_search(
            args["query"],
            top_k=args.get("top_k", 3),
            full_content=args.get("full_content", False),
        )
    if name == "list_tables":
        return run_list_tables(args.get("table_name"))
    if name == "list_documents":
        return run_list_documents()
    if name == "Skill":
        return run_skill(args["skill"])
    return f"未知工具：{name}"


# =============================================================================
# Agent 循环
# =============================================================================

def agent_loop(messages: list) -> list:
    """
    核心 Agent 循环（与 v1/v4 相同的模式）。

    模型持续调用工具，直到 stop_reason 不再是 'tool_use'。
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        # 收集工具调用，打印文本输出
        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
            if block.type == "tool_use":
                tool_calls.append(block)

        # 无工具调用 = 任务完成
        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return messages

        # 执行工具并收集结果
        results = []
        for tc in tool_calls:
            print(f"\n> {tc.name}: {tc.input}")
            output = execute_tool(tc.name, tc.input)
            preview = output[:200] + "..." if len(output) > 200 else output
            print(f"  {preview}")

            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": output,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# 主交互循环
# =============================================================================

def main():
    """
    交互式工业数据检索助手。

    支持多轮对话，维护对话历史（上下文记忆）。
    """
    print("=" * 60)
    print("工业设备管理系统 - 智能检索助手")
    print("=" * 60)
    print(f"数据库：{DB_PATH}")
    print(f"文档库：{DOCS_DIR}（{len(SAMPLE_DOCUMENTS)} 个文档）")
    print("\n示例查询：")
    print("  - CNC001设备当前状态如何？")
    print("  - 查询所有处于故障状态的设备")
    print("  - E003故障码的处理方法是什么？")
    print("  - 液压压力机紧急停机的操作流程")
    print("  - 哪些零件库存不足需要补货？")
    print("\n输入 'exit' 退出\n")

    history = []

    while True:
        try:
            user_input = input("技术员: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出系统。")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q", "退出"):
            print("退出系统。")
            break

        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"错误：{e}")

        print()


if __name__ == "__main__":
    main()
