"""Permitted-development Folium maps with honest segment-aware geometry."""
from pathlib import Path
from html import escape
import numpy as np
import folium
from src.hmm.adapter import route_polylines


def create_trip_map(output,trip_dir,destination,partition='student_test',max_network_arcs=2000,max_gps_points=1500):
    if partition not in {'train','student_test','synthetic'}:
        raise ValueError('Reference maps are limited to permitted development partitions')
    graph,obs=output['graph'],output['observations']
    valid=obs.loc[obs.coordinate_valid,['lat','lon']].to_numpy()
    center=np.mean(valid,axis=0).tolist() if len(valid) else [0,0]
    m=folium.Map(location=center,zoom_start=14,control_scale=True)
    network=folium.FeatureGroup(name='Road network (display sample)',show=False).add_to(m)
    ids=np.unique(np.linspace(0,len(graph.arc_from)-1,min(max_network_arcs,len(graph.arc_from)),dtype=int)) if len(graph.arc_from) else []
    for arc in ids:
        ll=graph.node_lonlat[[graph.arc_from[arc],graph.arc_to[arc]]]
        folium.PolyLine(ll[:,::-1].tolist(),color='#888888',weight=1,opacity=.5).add_to(network)
    gps=folium.FeatureGroup(name='GPS observations').add_to(m)
    stride=max(1,int(np.ceil(len(valid)/max_gps_points)))
    for lat,lon in valid[::stride]:
        folium.CircleMarker([lat,lon],radius=2,color='#e08019',fill=True,fill_opacity=.65).add_to(gps)
    for name,key,color in [('HMM prediction','route_hmm','#2463eb'),('Classifier prediction','route_classifier','#be37b8')]:
        group=folium.FeatureGroup(name=name).add_to(m)
        for line in output[key]:
            folium.PolyLine(line,color=color,weight=4,opacity=.8).add_to(group)
    # Explicit offline-reference layer; never imported by the prediction pipeline.
    reference=Path(trip_dir)/f'{Path(trip_dir).name}.route'
    if reference.is_file():
        from src.data.dataset_loader import load_route
        route=load_route(trip_dir,len(graph.arc_from))
        group=folium.FeatureGroup(name='Reference route (approximate)').add_to(m)
        for line in route_polylines([route],graph):
            folium.PolyLine(line,color='#008577',weight=5,opacity=.65).add_to(group)
    if len(valid):
        m.fit_bounds([valid.min(axis=0).tolist(),valid.max(axis=0).tolist()])
    title=f'PathSnap | {partition} | {escape(str(Path(trip_dir).name))}'
    legend=f'''<div style="position:fixed;top:10px;left:55px;z-index:9999;background:white;padding:12px;border:1px solid #bbb;font:13px Arial">
    <b>{title}</b><br>Development GB baseline / untuned decoder<br>
    <span style="color:#e08019">●</span> GPS &nbsp; <span style="color:#008577">━</span> Reference (approximate)<br>
    <span style="color:#2463eb">━</span> HMM &nbsp; <span style="color:#be37b8">━</span> Classifier<br>
    Display: {len(valid[::stride])}/{len(valid)} GPS, {len(ids)}/{len(graph.arc_from)} background arcs.<br>Inference uses all points/candidates; gaps are not joined.</div>'''
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    m.save(str(destination))
    return destination
