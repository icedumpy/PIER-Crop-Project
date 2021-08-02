import geopandas as gpd
#%%
gdf = gpd.read_file(r"F:\CROP-PIER\CROP-WORK\!common_shapefiles\thailand\thailand-tambon.shp")
#%%
list_p = ["สุรินทร์", "ร้อยเอ็ด", "นครสวรรค์", "สกลนคร", "อุทัยธานี", "นครราชสีมา", "ขอนแก่น"]
list_a = ["ท่าตูม", "ปทุมรัตต์", "ท่าตะโก", "วานรนิวาส", "สว่างอารมณ์", "ขามสะแกแสง", "แวงใหญ่"]
list_t = ["บะ", "ดอกล้ำ", "ท่าตะโก", "ขัวก่าย", "ไผ่เขียว", "ขามสะแกแสง", "โนนสะอาด"]
list_pat = [p+a+t for p, a, t in zip(list_p, list_a, list_t)]
#%%
gdf = gdf[(gdf["ADM1_TH"]+gdf["ADM2_TH"]+gdf["ADM3_TH"]).isin(list_pat)]
gdf[["ADM1_TH", "ADM2_TH", "ADM3_TH"]]
gdf.to_file(r"F:\CROP-PIER\CROP-WORK\Presentation\20210722\highlight.shp")