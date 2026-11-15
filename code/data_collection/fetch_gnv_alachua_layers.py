"""下载盖恩斯维尔市（Gainesville）与阿拉楚阿县（Alachua County, FL，FIPS 12001）的公开地理、人口与增长数据，
以及县 E911 账户公开发布的车辆级出动记录。

来源（全部免密钥，2026-10-21 逐个核实过地址与字段）：
  - 盖恩斯维尔市 ArcGIS Online（org Zzhtlau4ccHkQgTu，"City of Gainesville, FL"）：GFR 消防站、城市边界、兼并历史、开发项目
  - 阿拉楚阿县 GIS 服务器 maps.alachuacounty.us/server（Hosted 文件夹）：县公共安全部（ACDPS）消防站、消防与急救响应区、县界、市镇边界
  - 阿拉楚阿县 ArcGIS Online（org MiBZ4u97DWldovjI，"Alachua County Board of County Commissioners"，E911 人员账户发布，未写说明与许可）：
      FY2025（2024-10 至 2025-09）GFR + ACFR 全部车辆出动，含派车、出发、到场、离场、清场时间与派车时车辆位置；
      FY2024、FY2025 GFR 与 ACFR 之间的自动互助（AAA）出动
  - 盖恩斯维尔市 Socrata 开放数据 data.cityofgainesville.org：GFR + ACFR 首出动区（2020）、建筑许可（2011-01 至 2023-02）
  - 美国人口普查局：TIGERweb 2020 普查街区（POP100 人口、HU100 住房、UR 城乡标志）、2020 城市化地区、建筑许可调查（BPS）年度文件
  - 佛罗里达大学 BEBR 县人口预测（2026 版：2025 年估计 + 2030–2050 低/中/高）
  - OpenStreetMap（osmnx）全县可行车路网
  - 人口估计（PEP）直接取自 data/raw/growth 里已下载的两个 CSV，只做筛选与转长表，不重新下载

隐私：出动记录里的门牌地址列（PRIVATE_FIELDS）下载后立即删除，不落盘；坐标保留（分析需要）。这些记录不得逐点公开发布。

输出：data/raw/gnv_alachua/ 下每个数据集一个文件（扩展名决定格式：.gpkg 图层、.parquet 大表、.csv 小表、.graphml 路网），
以及 README.md（文件、行数、来源地址、许可、下载时间、说明）。

用法：
  python3 -I code/data_collection/fetch_gnv_alachua_layers.py                      # 下载清单中尚未下载的全部
  python3 -I code/data_collection/fetch_gnv_alachua_layers.py gfr_stations_2022.gpkg  # 只（重新）下载一个（名字见清单第一列）

只依赖官方包：requests、pandas、geopandas、osmnx（读 xlsx 用 pandas 自带的 openpyxl 引擎）。
"""
import datetime as dt
import io
import pathlib
import sys
import time

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests

# ---- 常量 ----
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gnv_alachua"
CITY_AGOL = "https://services2.arcgis.com/Zzhtlau4ccHkQgTu/arcgis/rest/services"
COUNTY_GIS = "https://maps.alachuacounty.us/server/rest/services/Hosted"
COUNTY_AGOL = "https://services1.arcgis.com/MiBZ4u97DWldovjI/arcgis/rest/services"
TIGERWEB = "https://tigerweb.geo.census.gov/arcgis/rest/services"
CITY_SOCRATA = "https://data.cityofgainesville.org"
BPS_URLS = {"county": "https://www2.census.gov/econ/bps/County/co{year}a.txt",
            "place": "https://www2.census.gov/econ/bps/Place/South%20Region/so{year}a.txt"}
BPS_YEARS = range(2000, 2026)
BEBR_URL = "https://bebr.ufl.edu/wp-content/uploads/2026/02/projections_2026.xlsx"
OSM_PLACE = "Alachua County, Florida, USA"
OVERPASS_URL = "https://z.overpass-api.de/api"                    # overpass-api.de 的 gall 节点；osmnx 会把主机名钉到一个 IP，
                                                                 # 2026-10-21 主域名解析到的 lambert 节点拒绝连接
ALACHUA_BBOX = (-82.66, 29.41, -82.05, 29.95)                     # 经纬度外包框，只用于取城市化地区
PRIVATE_FIELDS = ["Address"]                                      # 出动记录的门牌地址，不落盘
ARCGIS_PAGE = 2000
SOCRATA_PAGE = 50_000
RESIDENTIAL_NEW_PERMITS = (                                       # Gainesville 建筑许可里的"新建住宅"
    "type='Building Permit' AND (subtype='Manufactured Home (HUD)' OR (subtype='New Construction' AND "
    "classification in ('Residential Single Family','Residential Duplex','Apartments','Condominiums','Manufactured Buildings')))")
PUBLIC_DOMAIN = "Public Domain（数据集页面标注）"
NOT_STATED = "未声明"
US_GOV = "美国政府作品（公有领域）"

# (文件名, 来源类型, 地址, 附加参数（where 字符串 / 外包框 / None）, 许可, 说明)
LAYERS_TO_DOWNLOAD = [
    # (a) 消防站
    ("gfr_stations_2022.gpkg", "arcgis", f"{CITY_AGOL}/GFR_Stations/FeatureServer/0", None,
     "未声明", "GFR 消防站 8 个（1–5、7、8、9 号；无 6 号机场站），2022-06 最后编辑"),
    ("fire_stations_gfr_acfr_municipal_2020.gpkg", "arcgis", f"{CITY_AGOL}/Public_Safety_Layers/FeatureServer/1", None,
     NOT_STATED, "GFR、ACFR 与各市镇消防站 30 个，2020-12 快照（与 Socrata 7ang-kwek / gqcc-jpwj 同源）"),
    ("acdps_fire_ems_stations_2025.gpkg", "arcgis", f"{COUNTY_GIS}/ACDPS_fire_stations/FeatureServer/0", None,
     NOT_STATED, "县公共安全部的消防/急救站 28 个，2025-05 编辑；type 区分 Fire / EMS / Fire-EMS；缺 GFR 8、9 号"),
    # (b) 边界与响应区
    ("gainesville_city_limits_2026.gpkg", "arcgis", f"{CITY_AGOL}/Gainesville_Boundary/FeatureServer/0", None,
     "仅供参考、不保证精度（条目 licenseInfo）", "盖恩斯维尔市界，2026-08-11 更新"),
    ("gainesville_annexation_history.gpkg", "arcgis", f"{CITY_AGOL}/Annexation_History/FeatureServer/17", None,
     NOT_STATED, "历次兼并地块，含 ANNEX_YEAR 与类型"),
    ("alachua_county_boundary.gpkg", "arcgis", f"{COUNTY_GIS}/Alachua_County_Boundary/FeatureServer/0", None,
     NOT_STATED, "县界 1 个多边形"),
    ("alachua_municipalities.gpkg", "arcgis", f"{COUNTY_GIS}/Municipalities/FeatureServer/0", None,
     NOT_STATED, "9 个建制市镇边界，2026-02 编辑"),
    ("acdps_fire_response_areas_2025.gpkg", "arcgis", f"{COUNTY_GIS}/ACDPS_fire_resp_areas/FeatureServer/0", None,
     NOT_STATED, "县消防响应区 18 个，district=站号（0 似为 GFR 市区），2025-05 编辑"),
    ("acdps_medic_response_areas_2025.gpkg", "arcgis", f"{COUNTY_GIS}/ACDPS_medic_resp_areas/FeatureServer/0", None,
     NOT_STATED, "县急救（Rescue）响应区 8 个，2025-05 编辑"),
    ("gfr_acfr_first_due_2020.gpkg", "socrata_geo", "m7cy-v78q", None,
     NOT_STATED, "GFR 与 ACFR 首出动区叠加（14 块，name_1 为站名），2020-11；不含 GFR 9 号站"),
    # (c)(d) 普查街区与城市化地区
    ("census2020_blocks_alachua.gpkg", "arcgis", f"{TIGERWEB}/Census2020/Tracts_Blocks/MapServer/2",
     "STATE='12' AND COUNTY='001'", US_GOV, "2020 普查街区，POP100 人口、HU100 住房、UR=U/R 城乡（2020 标准）"),
    ("census2020_urban_areas_alachua_bbox.gpkg", "arcgis", f"{TIGERWEB}/TIGERweb/Urban/MapServer/0", ALACHUA_BBOX,
     US_GOV, "与全县外包框相交的 2020 城市化地区；只有 Gainesville UA (32167) 含本县街区"),
    # (e) 路网
    ("alachua_drive.graphml", "osm", OSM_PLACE, None,
     "ODbL（© OpenStreetMap contributors）", "osmnx 可行车路网，边上有 speed_kph 与 travel_time（自由流，未校准）"),
    ("alachua_e911_road_centerlines.gpkg", "arcgis", f"{COUNTY_AGOL}/Road_Centerline_Public_view/FeatureServer/1", None,
     NOT_STATED, "县 E911（NG911）道路中心线，含 ROADCLASS、ONEWAY、ESN_L/ESN_R（应急服务号，对应出动辖区），无限速"),
    # (f) 建筑许可与新增住房
    ("gainesville_permits_residential_new_2011_2023.gpkg", "socrata_geo", "p798-x3nx", RESIDENTIAL_NEW_PERMITS,
     PUBLIC_DOMAIN, "市建筑许可中的新建住宅（独栋、双拼、公寓、共管、预制房），无套数列（公寓一栋一证）；56 条无坐标；数据止于 2023-02"),
    ("gainesville_development_projects_2021_2026.gpkg", "arcgis",
     f"{CITY_AGOL}/Gainesville_Development_Projects_Updated/FeatureServer/9", None,
     NOT_STATED, "市开发项目申请（地块多边形），含 Units、Bedrooms、申请与签发日期"),
    ("census_bps_alachua_county_annual.csv", "bps_county", BPS_URLS["county"], None,
     US_GOV, "建筑许可调查：全县每年新批住宅栋数/套数/造价，按 1、2、3–4、5+ 套分组，2000–2025"),
    ("census_bps_alachua_places_annual.csv", "bps_place", BPS_URLS["place"], None,
     US_GOV, "建筑许可调查：县内各发证地（含县属非建制区）每年新批住宅，2000–2025；各地之和等于全县；"
     "Alachua 市 2023–2025 为 0（疑似未上报）；Archer、Hawthorne、Micanopy 2003 年后不单列"),
    # (g) 人口
    ("alachua_population_pep_2010_2024.csv", "pep_local", "data/raw/growth", None,
     US_GOV, "PEP 7 月 1 日人口：全县、9 个市镇与非建制区，2010–2019 跨普查修订版 + 2020–2024 Vintage 2024"),
    ("bebr_alachua_projections_2026.csv", "bebr", BEBR_URL, None,
     NOT_STATED, "BEBR：2025-04-01 估计与 2030–2050 低/中/高预测（原始 xlsx 同目录保存）"),
    # (h) 出动记录（县 E911 账户公开发布）
    ("acfr_gfr_unit_responses_fy2025.parquet", "arcgis", f"{COUNTY_AGOL}/FY25_ALL_UNIT_RESPONSES/FeatureServer/38", None,
     NOT_STATED, "FY2025 车辆级出动 10.5 万行、5.5 万起事件，含派车/出发/到场/离场/清场时间与派车时车辆位置；"
     "Jurisdiction 是事件所在辖区（GFR 区 5.8 万、ACFR 区 4.7 万），不是车辆所属机构，归属要按车号判断；地址列已删"),
    ("acfr_gfr_automatic_aid_fy2024.parquet", "arcgis", f"{COUNTY_AGOL}/AAA_FY24_ALL/FeatureServer/0", None,
     NOT_STATED, "FY2024 GFR 与 ACFR 自动互助出动（Juri=事件辖区，Radio_Name=对方机构出动的车辆），只有日期；地址列已删"),
    ("acfr_gfr_automatic_aid_fy2025.parquet", "arcgis", f"{COUNTY_AGOL}/AAA__FY25_ALL/FeatureServer/43", None,
     NOT_STATED, "FY2025 自动互助出动；地址列已删"),
]


def request_with_retries(url: str, params: dict = None, attempts: int = 4, verbose: bool = False) -> requests.Response:
    """GET 一个地址，失败时等 2、4、8 秒重试，最后一次仍失败就抛出异常。

    输入 url: 地址，str，例: "https://www2.census.gov/econ/bps/County/co2024a.txt"
    输入 params: 查询参数，dict，例: {"where": "1=1", "f": "geojson"}
    输出: requests.Response（状态码 200），例: <Response [200]>
    """
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=300, headers={"User-Agent": "Mozilla/5.0 (research)"})
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            if attempt == attempts:
                raise
            if verbose:
                print(f"[request_with_retries] 第 {attempt} 次失败（{error}），{2 ** attempt} 秒后重试")
            time.sleep(2 ** attempt)


def download_arcgis_layer(layer_url: str, extra=None, verbose: bool = False) -> gpd.GeoDataFrame:
    """分页读取一个 ArcGIS 图层（FeatureServer 或 MapServer）的全部要素，坐标转为 WGS84。

    输入 layer_url: 图层地址，str，例: "https://maps.alachuacounty.us/server/rest/services/Hosted/ACDPS_fire_stations/FeatureServer/0"
    输入 extra: None、where 字符串（例 "STATE='12' AND COUNTY='001'"）或经纬度外包框 tuple（例 (-82.66, 29.41, -82.05, 29.95)）
    输出: GeoDataFrame，crs=EPSG:4326，例: ACDPS 消防站 shape=(28, 17)，列含 facility、type、jurisdicti、geometry
    """
    params = {"where": extra if isinstance(extra, str) else "1=1", "outFields": "*", "outSR": 4326,
              "f": "geojson", "resultRecordCount": ARCGIS_PAGE}
    if isinstance(extra, tuple):
        params.update(geometry=",".join(map(str, extra)), geometryType="esriGeometryEnvelope", inSR=4326,
                      spatialRel="esriSpatialRelIntersects")
    features = []
    while True:
        page = request_with_retries(f"{layer_url}/query", {**params, "resultOffset": len(features)}, verbose=verbose).json()
        if "error" in page:
            raise RuntimeError(f"{layer_url}: {page['error']}")
        batch = page.get("features", [])
        features.extend(batch)
        exceeded = page.get("exceededTransferLimit") or page.get("properties", {}).get("exceededTransferLimit")
        if verbose:
            print(f"[download_arcgis_layer] {layer_url.split('/services/')[-1]}: 已取 {len(features):,} 个要素", end="\r")
        if not batch or (not exceeded and len(batch) < ARCGIS_PAGE):
            break
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if verbose:
        print(f"\n[download_arcgis_layer] 共 {len(frame):,} 个要素，列: {list(frame.columns)[:12]}")
    return frame


def download_socrata_geojson(dataset_id: str, where: str = None, verbose: bool = False) -> gpd.GeoDataFrame:
    """分页读取 data.cityofgainesville.org 上一个带几何的数据集（.geojson 端点），可带 SoQL 过滤。

    输入 dataset_id: Socrata 四四码，str，例: "m7cy-v78q"
    输入 where: SoQL 条件，str，例: "type='Building Permit' AND subtype='New Construction'"
    输出: GeoDataFrame，crs=EPSG:4326，例: 首出动区 shape=(14, 23)，列含 name_1、code_1、geometry
    """
    features = []
    while True:
        params = {"$limit": SOCRATA_PAGE, "$offset": len(features), "$order": ":id"}
        if where:
            params["$where"] = where
        batch = request_with_retries(f"{CITY_SOCRATA}/resource/{dataset_id}.geojson", params, verbose=verbose).json()["features"]
        features.extend(batch)
        if verbose:
            print(f"[download_socrata_geojson] {dataset_id}: 已取 {len(features):,} 个要素")
        if len(batch) < SOCRATA_PAGE:
            break
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def download_bps_alachua(url_pattern: str, level: str, verbose: bool = False) -> pd.DataFrame:
    """逐年读取普查局建筑许可调查（BPS）年度文件，只留佛罗里达州阿拉楚阿县（县码 001）的行。

    每个文件有两行表头；"Name" 列之后固定是 24 个数：先 12 个含插补的估计（1 套、2 套、3–4 套、5 套以上 ×
    栋数/套数/造价），再 12 个仅上报值（列名带 _reported）。分析用前 12 个。
    输入 url_pattern: 带 {year} 的地址，str，例: "https://www2.census.gov/econ/bps/County/co{year}a.txt"
    输入 level: "county" 或 "place"，str（县文件第 3 列是县码，地方文件第 4 列是县码）
    输出: DataFrame，例: level="county" 时 shape=(26, 26)，一行一年：
          year=2025, name="Alachua County", units_1_unit=689, units_5plus=2629, ...
    """
    county_column = {"county": 2, "place": 3}[level]
    groups = ["1_unit", "2_units", "3_4_units", "5plus"]
    value_names = [f"{measure}_{group}{suffix}" for suffix in ("", "_reported")
                   for group in groups for measure in ("bldgs", "units", "value")]
    pages = []
    for year in BPS_YEARS:
        text = request_with_retries(url_pattern.format(year=year), verbose=verbose).text
        name_at = [cell.strip() for cell in text.splitlines()[1].split(",")].index("Name")
        table = pd.read_csv(io.StringIO(text), skiprows=2, header=None, dtype=str).fillna("")
        table = table[(table[0].str.strip() == str(year)) & (table[1].str.strip() == "12")
                      & (table[county_column].str.strip() == "001")]
        page = table[list(range(name_at + 1, name_at + 25))].astype(int)
        page.columns = value_names
        page.insert(0, "name", table[name_at].str.strip())
        page.insert(0, "year", year)
        pages.append(page)
        if verbose:
            print(f"[download_bps_alachua] {level} {year}: {len(page)} 行，新批住宅 "
                  f"{page[[c for c in value_names[:12] if c.startswith('units_')]].to_numpy().sum():,} 套", end="\r")
    if verbose:
        print()
    return pd.concat(pages, ignore_index=True)


def build_alachua_population_table(growth_dir: pathlib.Path, verbose: bool = False) -> pd.DataFrame:
    """从已下载的两个 PEP 文件里取阿拉楚阿县（050 县、157 市镇及"Balance of"非建制区）的 7 月 1 日人口，转成长表。

    输入 growth_dir: 存放 PEP 文件的目录，Path，例: data/raw/growth
    输出: DataFrame，列 geography(str)、sumlev(str)、place_fips(str)、year(int)、population(int)、vintage(str)，
          例: 165 行；("Alachua County", "050", "00000", 2024, 291782, "V2024")
    """
    intercensal = pd.read_csv(growth_dir / "census_pep_subcounty_2010_2020_intercensal.csv", encoding="latin-1", dtype=str)
    vintage2024 = pd.read_csv(growth_dir / "census_pep_subcounty_2020_2024_florida.csv", encoding="latin-1", dtype=str)
    pieces = []
    for source, years, label in ((intercensal, range(2010, 2020), "2010-2020 intercensal"),
                                 (vintage2024, range(2020, 2025), "V2024")):
        rows = source[(source.STATE == "12") & (source.COUNTY == "001") & source.SUMLEV.isin(["050", "157"])]
        long = rows.melt(id_vars=["NAME", "SUMLEV", "PLACE"], value_vars=[f"POPESTIMATE{y}" for y in years],
                         var_name="year", value_name="population")
        long["year"] = long["year"].str[-4:].astype(int)
        long["vintage"] = label
        pieces.append(long)
    table = pd.concat(pieces, ignore_index=True).rename(columns={"NAME": "geography", "SUMLEV": "sumlev", "PLACE": "place_fips"})
    table["population"] = table["population"].astype(int)
    if verbose:
        county = table[table.sumlev == "050"].set_index("year")["population"]
        print(f"[build_alachua_population_table] {len(table)} 行；全县 2010 年 {county[2010]:,} → 2024 年 {county[2024]:,}")
    return table


def download_bebr_alachua(url: str, verbose: bool = False) -> pd.DataFrame:
    """下载 BEBR 县人口预测 xlsx（原样存一份），取出阿拉楚阿县的估计与低/中/高预测。

    输入 url: xlsx 地址，str，例: "https://bebr.ufl.edu/wp-content/uploads/2026/02/projections_2026.xlsx"
    输出: DataFrame，列 series(str)、year(int)、population(int)，例: 16 行；("estimate", 2025, 298485)、("Medium", 2050, 345200)
    """
    content = request_with_retries(url, verbose=verbose).content
    (OUTPUT_DIR / url.split("/")[-1]).write_bytes(content)
    sheet = pd.read_excel(io.BytesIO(content), header=None)
    years = [int(float(y)) for y in sheet.iloc[1, 3:8]]                     # 第 2 行是预测年份
    first = sheet.index[sheet[0].astype(str).str.strip() == "ALACHUA"][0]
    rows = [("estimate", int(str(sheet.iloc[1, 1])[-4:]), int(sheet.iloc[first, 1]))]
    for offset in (1, 2, 3):                                                # 下面三行依次是 Low、Medium、High
        series = str(sheet.iloc[first + offset, 0]).strip()
        rows += [(series, year, int(value)) for year, value in zip(years, sheet.iloc[first + offset, 3:8])]
    table = pd.DataFrame(rows, columns=["series", "year", "population"])
    if verbose:
        medium_2050 = table[(table.series == "Medium") & (table.year == 2050)]["population"].iloc[0]
        print(f"[download_bebr_alachua] {len(table)} 行；{rows[0][1]} 估计 {rows[0][2]:,}，2050 中方案 {medium_2050:,}")
    return table


def download_drive_graph(place_name: str, path: pathlib.Path, verbose: bool = False):
    """从 OpenStreetMap 下载可行车路网，按 maxspeed（缺失时按道路等级均值）补车速与边行驶秒数，存成 GraphML。

    输入 place_name: 地名，str，例: "Alachua County, Florida, USA"
    输入 path: 输出路径，Path，例: data/raw/gnv_alachua/alachua_drive.graphml
    输出: networkx.MultiDiGraph，边上有 length(m)、speed_kph、travel_time(s)，例: 约 3 万个节点、8 万条边（以实际为准）
    """
    ox.settings.overpass_url = OVERPASS_URL
    for attempt in range(1, 5):                          # Overpass 常满载（502 / 拒绝连接），等 2、4、8 分钟重试
        try:
            graph = ox.graph_from_place(place_name, network_type="drive")
            break
        except Exception as error:
            if attempt == 4:
                raise
            if verbose:
                print(f"[download_drive_graph] 第 {attempt} 次失败（{str(error)[:120]}），{2 ** attempt} 分钟后重试")
            time.sleep(60 * 2 ** attempt)
    graph = ox.routing.add_edge_travel_times(ox.routing.add_edge_speeds(graph))
    ox.save_graphml(graph, path)
    if verbose:
        print(f"[download_drive_graph] {graph.number_of_nodes():,} 个节点，{graph.number_of_edges():,} 条边 → {path.name}")
    return graph


def download_and_save(file_name: str, kind: str, address: str, extra, verbose: bool = False) -> pathlib.Path:
    """按来源类型下载一个数据集，并按文件扩展名保存（.gpkg / .parquet / .csv；路网由 download_drive_graph 自己存）。

    存 .parquet / .csv 时去掉几何列（坐标另有列），并删掉 PRIVATE_FIELDS 中的列。
    输入: LAYERS_TO_DOWNLOAD 的前四列，例: ("alachua_county_boundary.gpkg", "arcgis", ".../FeatureServer/0", None)
    输出: 写出的路径，例: data/raw/gnv_alachua/alachua_county_boundary.gpkg
    """
    path = OUTPUT_DIR / file_name
    if kind == "osm":
        download_drive_graph(address, path, verbose=verbose)
        return path
    if kind == "arcgis":
        frame = download_arcgis_layer(address, extra, verbose=verbose)
    elif kind == "socrata_geo":
        frame = download_socrata_geojson(address, extra, verbose=verbose)
    elif kind in ("bps_county", "bps_place"):
        frame = download_bps_alachua(address, kind.split("_")[1], verbose=verbose)
    elif kind == "pep_local":
        frame = build_alachua_population_table(PROJECT_ROOT / address, verbose=verbose)
    elif kind == "bebr":
        frame = download_bebr_alachua(address, verbose=verbose)
    else:
        raise ValueError(f"未知来源类型: {kind}")
    frame = frame.drop(columns=[c for c in PRIVATE_FIELDS if c in frame.columns])
    if path.suffix == ".gpkg":                           # GeoPackage 自带 fid 主键，来源里的 fid 列改名，否则字符串 fid 写不进去
        frame = frame.rename(columns={c: f"source_{c.lower()}" for c in frame.columns if c.lower() == "fid"})
        frame.to_file(path, driver="GPKG")
    else:
        frame = pd.DataFrame(frame.drop(columns="geometry", errors="ignore"))
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
    if verbose:
        print(f"[download_and_save] {file_name}: {len(frame):,} 行，{path.stat().st_size / 1e6:.1f} MB")
    return path


def count_rows(path: pathlib.Path) -> str:
    """读回一个已保存的文件，返回行数（路网返回节点与边数），给 README 用。

    输入 path: 文件路径，Path，例: data/raw/gnv_alachua/census2020_blocks_alachua.gpkg
    输出: str，例: "5,897"；路网例: "31,000 节点 / 80,000 边"
    """
    if path.suffix == ".graphml":
        graph = ox.load_graphml(path)
        return f"{graph.number_of_nodes():,} 节点 / {graph.number_of_edges():,} 边"
    reader = {".gpkg": gpd.read_file, ".parquet": pd.read_parquet, ".csv": pd.read_csv}[path.suffix]
    return f"{len(reader(path)):,}"


def write_readme(verbose: bool = False) -> pathlib.Path:
    """写 README.md：清单里每个已存在的文件一行（行数、来源、许可、下载时间=文件修改时间、说明），并列出缺失的文件。

    输入: 无（读 LAYERS_TO_DOWNLOAD 与 OUTPUT_DIR）
    输出: README 路径，例: data/raw/gnv_alachua/README.md
    """
    lines = ["# 盖恩斯维尔与阿拉楚阿县：公开地理、人口、增长与出动数据", "",
             "- 下载脚本：code/data_collection/fetch_gnv_alachua_layers.py",
             f"- README 生成时间：{dt.datetime.now().isoformat(timespec='seconds')}；每个文件的下载时间见表中一列",
             "- 坐标：所有 .gpkg 均为 EPSG:4326；出动表的 GIS_LAT/GIS_LONG（或 Lat/Long）为十进制度，Latitude/Longitude 为乘 1e6 的整数且经度去了负号",
             "- 出动表时间列带 +00:00 后缀，但 FY2025 首末记录为 2024-10-01 00:06 与 2025-09-30 23:53，说明实为当地钟面时间（America/New_York），不要再做时区换算",
             "- 隐私：出动表的门牌地址列下载时已删除；EMS 记录含问题类型与坐标，只做汇总展示，不逐点公开",
             "- 出动表来自县 E911 人员账户的公开条目，无说明、无许可；正式使用前请经 Mitterer 教授向 ACFR / 联合通讯中心确认", "",
             "| 文件 | 行数 | 来源 | 许可 | 下载时间 | 说明 |", "|---|---|---|---|---|---|"]
    missing = []
    for file_name, kind, address, extra, license_text, note in LAYERS_TO_DOWNLOAD:
        path = OUTPUT_DIR / file_name
        if not path.exists():
            missing.append(file_name)
            continue
        source = address if kind not in ("bps_county", "bps_place") else address.replace("{year}", "{2000..2025}")
        if kind == "socrata_geo":
            source = f"{CITY_SOCRATA}/resource/{address}.geojson"
        downloaded = dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes")
        lines.append(f"| {file_name} | {count_rows(path)} | {source} | {license_text} | {downloaded} | {note} |")
    lines += ["", f"另存原始文件：{BEBR_URL.split('/')[-1]}（BEBR 原表，全州各县）"]
    if missing:
        lines += ["", "未下载成功：" + "、".join(missing)]
    path = OUTPUT_DIR / "README.md"
    path.write_text("\n".join(lines) + "\n")
    if verbose:
        print(f"[write_readme] {path}（缺 {len(missing)} 个）")
    return path


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for file_name, kind, address, extra, _license, _note in LAYERS_TO_DOWNLOAD:
        if (only and file_name not in only) or (not only and (OUTPUT_DIR / file_name).exists()):
            continue
        print(f"== {file_name}")
        try:
            download_and_save(file_name, kind, address, extra, verbose=True)
        except Exception as error:                       # 单个失败不影响其余；README 末尾会列出缺失的文件
            print(f"!! {file_name} 下载失败: {error}")
            (OUTPUT_DIR / file_name).unlink(missing_ok=True)  # 删掉写了一半的文件，下次运行才会重下
    write_readme(verbose=True)
