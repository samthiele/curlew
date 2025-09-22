"""
Functions for performing common IO operations.
"""

import os
import numpy as np

def saveOBJ(filename, xyz, rgb, faces):
    """
    Writes a mesh to an OBJ file.

    Parameters
    ---------------
    filename : str
        Output file path.
    xyz : np.ndarray | list
        List of vertex positions [(x, y, z), ...]
    rgb: np.ndarray | list
        List of vertex colors [(r, g, b), ...] or [(r, g, b, a), ...] in [0, 1] or [0, 255]
    faces: np.ndarray | list
        List of faces [(i1, i2, i3), ...] with 0-based indices
    """
    with open(filename, 'w') as f:
        f.write("# OBJ file with vertex colors (stored in comments)\n")

        for i, v in enumerate(xyz):
            x, y, z = v
            color_str = ""
            if (rgb is not None) and i < len(rgb):
                color = rgb[i]
                # Normalize to 0-1 if in 0-255
                if max(color) > 1:
                    color = [c / 255.0 for c in color[:3]]
                else:
                    color = color[:3]
                color_str = " # color {:.4f} {:.4f} {:.4f}".format(*color)
            f.write("v {:.6f} {:.6f} {:.6f}{}\n".format(x, y, z, color_str))

        for face in faces:
            # OBJ format is 1-indexed
            f.write("f {}\n".format(' '.join(str(i + 1) for i in face)))

def savePLY(path, xyz, rgb=None, normals=None, attr=None, names=None, faces=None):
    """
    Write a point cloud and associated RGB and scalar fields to .ply.

    Parameters
    ---------------
    Path : str
        File path for the created (or overwritten) .ply file
    xyz : np.ndarray
        Array of xyz points to add to the PLY file
    rgb : np.ndarray
        Array of 0-255 RGB values associated with these points, or None.
    normals : np.ndarray
        Array of normal vectors associated with each point, or None.
    attr : np.ndarray
        Array of float32 values associated with these points, or None
    attr_names : list 
        List containing names for each of the passed attributes, or None.
    faces : np.ndarray, optional
        Array of triangular faces (F x 3) with vertex indices, or None.
    """

    # make directories if need be
    os.makedirs(os.path.dirname( path ), exist_ok=True )

    try:
        from plyfile import PlyData, PlyElement
    except:
        assert False, "Please install plyfile (`pip install plyfile`) to export to PLY."

    sfmt='f4' # use float32 precision

    # create structured data arrays and derived PlyElements
    vertex = np.array(list(zip(xyz[:, 0], xyz[:, 1], xyz[:, 2])),
                      dtype=[('x', 'double'), ('y', 'double'), ('z', 'double')])
    ply = [PlyElement.describe(vertex, 'vertices')]

    # create RGB elements
    if rgb is not None:
        if (np.max(rgb) <= 1):
            irgb = np.clip((rgb * 255),0,255).astype(np.uint8)
        else:
            irgb = np.clip(rgb,0,255).astype(np.uint8)

        # convert to structured arrays and create elements
        irgb = np.array(list(zip(irgb[:, 0], irgb[:, 1], irgb[:, 2])),
                        dtype=[('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        ply.append(PlyElement.describe(irgb, 'color'))  # create ply elements

    # normal vectors
    if normals is not None:
        # convert to structured arrays
        norm = np.array(list(zip(normals[:, 0], normals[:, 1], normals[:, 2])),
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply.append(PlyElement.describe(norm, 'normals'))  # create ply elements

    # attributes
    if attr is not None:
        if names is None:
            names = ["SF%d"%(i+1) for i in range(attr.shape[-1])]
        
        # map scalar fields to required type and build data arrays
        data = attr.astype(np.float32)
        for b in range(data.shape[-1]):
            n = names[b].strip().replace(' ', '_') #remove spaces from n
            if 'scalar' in n: #name already includes 'scalar'?
                ply.append(PlyElement.describe(data[:, b].astype([('%s' % n, sfmt)]), '%s' % n))
            else: #otherwise prepend it (so CloudCompare recognises this as a scalar field).
                ply.append(PlyElement.describe(data[:, b].astype([('scalar_%s' % n, sfmt)]), 'scalar_%s' % n))
    
    # Append faces if present
    if faces is not None and len(faces) > 0:
        faces = np.asarray(faces, dtype=np.int32)
        face_data = np.array(
            [(list(face),) for face in faces],
            dtype=[('vertex_indices', 'i4', (3,))]
        )
        ply.append(PlyElement.describe(face_data, 'face'))
    
    PlyData(ply).write(path) # and, finally, write everything :-) 

def loadPLY(path):
    """
    Loads a PLY file from the specified path.
    """
    try:
        from plyfile import PlyData, PlyElement
    except:
        assert False, "Please install plyfile (pip install plyfile) to load PLY."
    data = PlyData.read(path) # load file!

    # extract data
    xyz = None
    rgb = None
    norm = None
    faces = None
    scalar = []
    scalar_names = []
    
    for e in data.elements:
        if 'face' in e.name.lower():
            faces = np.vstack( e['vertex_indices'])
        if 'vert' in e.name.lower():  # vertex data
            xyz = np.array([e['x'], e['y'], e['z']]).T
            if len(e.properties) > 3:  # vertices have more than just position
                names = e.data.dtype.names
                # colour?
                if 'red' in names and 'green' in names and 'blue' in names:
                    rgb = np.array([e['red'], e['green'], e['blue']], dtype=e['red'].dtype).T
                # normals?
                if 'nx' in names and 'ny' in names and 'nz' in names:
                    norm = np.array([e['nx'], e['ny'], e['nz']], dtype=e['nx'].dtype).T
                # load others as scalar
                mask = ['red', 'green', 'blue', 'nx', 'ny', 'nz', 'x', 'y', 'z']
                for n in names:
                    if not n in mask:
                        scalar_names.append(n)
                        scalar.append(e[n])
        elif 'color' in e.name.lower():  # rgb data
            rgb = np.array([e['r'], e['g'], e['b']], dtype=e['r'].dtype).T
        elif 'normals' in e.name.lower():  # normal data
            norm = np.array([e['x'], e['y'], e['z']], dtype=e['z'].dtype).T
        else:  # scalar data
            scalar_names.append(e.properties[0].name.strip().replace('scalar_',''))
            scalar.append(np.array(e[e.properties[0].name], dtype=e[e.properties[0].name].dtype))
    if len(scalar) > 0:
        scalar = np.vstack(scalar).T
    assert (not xyz is None) and (xyz.shape[0] > 0), "Error - PLY contains no geometry?"

    # TODO - also load faces if present
    
    # return everything needed
    out = dict( xyz = xyz, faces=faces, rgb=rgb, normals=norm, 
                attr=scalar, names=scalar_names )
    if len(scalar) == 0:
        del out['attr']
        del out['names']
    if rgb is None:
        del out['rgb']
    if norm is None:
        del out['normals']
    if faces is None:
        del out['faces']
    return out