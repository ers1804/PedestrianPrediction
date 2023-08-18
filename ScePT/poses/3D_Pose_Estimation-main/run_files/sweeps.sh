# Sweep Lifting with 2D labels
python main.py --tune waymo_2d_labels_supervised --model_type SimpleLiftingModel
# resume this sweep by passing id (not used yet)
python main.py --tune waymo_2d_labels_supervised --model_type SimpleLiftingModel --sweep_id t5sqzvjk

# Sweep Lifting model with 3D projections
python main.py --tune waymo_3d_2d_projections_supervised --device 1 --model_type SimpleLiftingModel
# resume this sweep by passing the id
python main.py --tune waymo_3d_2d_projections_supervised --model_type SimpleLiftingModel --sweep_id ows5hu6l


# Sweep Fusion with 3D projections
python main.py --tune waymo_3d_2d_projections_supervised
# resume this sweep by passing the id
python main.py --tune waymo_3d_2d_projections_supervised --sweep_id ij1izc82


# Sweep Fusion with 2D labels
python main.py --tune waymo_2d_labels_supervised
# resume this sweep by passing id (not used yet)
python main.py --tune waymo_2d_labels_supervised --sweep_id scwqbb2a
