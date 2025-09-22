"""
A lightweight visualisation tool for jupyter notebooks using `ipywidgets` and `pythreejs`. 
This will likely be depricated when we find a better option... (that is lighter than VTK).
"""
import numpy as np

try:
    import ipywidgets
except:
    assert False, "Please install `ipywidgets` to use DataScreen."

try:
    import pythreejs as p3s
except:
    assert False, "Please install `pythreejs` to use DataScreen."
from IPython.display import display

class DataScreen:
    def __init__( self, scale_factor=None, mode = "JUPYTER", **kwds ):
        self.sf = scale_factor
        self.center = None
        self.mode = mode
        self.__update_settings(kwds)
        self._light = p3s.DirectionalLight(color='white', position=[0, 0, 1], intensity=0.6)
        self._light2 = p3s.AmbientLight(intensity=0.5)
        self._cam = p3s.PerspectiveCamera(position=[0, 0, 10], lookAt=[0, 0, 0], fov=self.__s["fov"],
                                     aspect=self.__s["width"]/self.__s["height"], children=[self._light])
        self._cam.up = (0,0,1) # set up direction to z-axis (important for controller)

        self._orbit = p3s.OrbitControls(controlling=self._cam)
        #self._orbit = p3s.TrackballControls(controlling=self._cam)
        self._scene = p3s.Scene(children=[self._cam, self._light2], background=self.__s["background"])#"#4c4c80"
        self._renderer = p3s.Renderer(camera=self._cam, scene = self._scene, controls=[self._orbit],
                    width=self.__s["width"], height=self.__s["height"], antialias=self.__s["antialias"])

        self._objects = {}
        self._cnt = 0
        self._centroid = None

        # init GUI variables (these will be added if the corresponding geometry type is added)
        self._size_slider = None
        self._clip_slider = {}
        self._picker = None
        self.pickedPoints = []
        self.pickedIDs = []

    def __update_settings(self, settings={}):
        sett = {"width": 600, "height": 300, "antialias": True, "scale": 1.5, "background": "#ffffff",
                "fov": 45}
        for k in settings:
            sett[k] = settings[k]
        self.__s = sett

    def __add_object(self, name, obj):
        self._objects[name] = obj
        self._cnt += 1
        self._scene.add(obj["mesh"])
        self.__update_view()
        if self.mode == "JUPYTER":
            return self._cnt - 1
        elif self.mode == "WEBSITE":
            return self
        
    def __update_view(self):
        if len(self._objects) == 0:
            return
        ma = np.zeros((len(self._objects), 3))
        mi = np.zeros((len(self._objects), 3))
        for r, obj in enumerate(self._objects):
            ma[r] = self._objects[obj]["max"]
            mi[r] = self._objects[obj]["min"]
        ma = np.max(ma, axis=0)
        mi = np.min(mi, axis=0)
        diag = np.linalg.norm(ma-mi)
        mean = ((ma - mi) / 2 + mi).tolist()
        self._centroid = mean # store as this is quite useful
        
        scale = self.__s["scale"] * (diag)
        self._orbit.target = mean
        #self._cam.lookAt(mean)
        self._cam.position = [mean[0], mean[1], mean[2]+scale]
        self._light.position = [mean[0], mean[1], mean[2]+scale]

        self._orbit.exec_three_obj_method('update')
        self._cam.exec_three_obj_method('updateProjectionMatrix')
    
    def __get_bbox(self, v):
        m = np.min(v, axis=0)
        M = np.max(v, axis=0)

        # Corners of the bounding box
        v_box = np.array([[m[0], m[1], m[2]], [M[0], m[1], m[2]], [M[0], M[1], m[2]], [m[0], M[1], m[2]],
                          [m[0], m[1], M[2]], [M[0], m[1], M[2]], [M[0], M[1], M[2]], [m[0], M[1], M[2]]])

        f_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [7, 3]], dtype=np.uint32)
        return v_box, f_box
    
    def __centerScale( self, xyz ):
        # get and apply center
        if self.center is None:
            self.center = np.mean(xyz, axis=0)
        xyz = xyz - self.center # remove center

        # get and apply scale factor
        if self.sf is None:
            self.sf = 10 / np.max( np.max(xyz, axis=0) - np.min(xyz, axis=0) )
        xyz = xyz * self.sf 

        return xyz
    
    def addMesh(self, name, verts, faces, rgb='green'):
        verts = self.__centerScale( verts )
        
        geometry = p3s.BufferGeometry(attributes=dict(index=p3s.BufferAttribute(faces.astype(np.uint32).ravel(), normalized=False), 
                                                    position=p3s.BufferAttribute(verts.astype(np.float32), normalized=False)))
        #geometry.exec_three_obj_method('computeFaceNormals')
        material = p3s.MeshStandardMaterial(color=rgb, side='DoubleSide', flat=True,
                                            roughness=0.5, metalness=0.25, reflectivity=1.0,)
        mesh = p3s.Mesh(geometry=geometry, material=material)

        obj = {
            'mesh' : mesh,
            'geometry' : geometry,
            'material' : material,
            'type' : "Mesh",
            'min' : np.min(verts, axis=0),
            'max' : np.max(verts, axis=0),
            'bounds' : self.__get_bbox(verts),
        }
        self.__add_object(name, obj)

    def addPoints(self, name, xyz, rgb=None):
        xyz = self.__centerScale( xyz )

        # get bounds
        bounds = self.__get_bbox(xyz)
        ma = np.max(xyz, axis=0)
        mi = np.min(xyz, axis=0)

        # build slider if needed
        if self._size_slider is None:
            self._size_slider = ipywidgets.FloatSlider(value=0.02, min=0.0, max=0.1, description='Point Size', step=0.01)
        ps = self._size_slider.value

        # build buffer attribute
        pts = p3s.BufferAttribute( array=(xyz).astype(np.float32) )
        if rgb is not None:
            if isinstance(rgb, str): # constant colour
                geometry = p3s.BufferGeometry( attributes={'position': pts })
                material = p3s.PointsMaterial(color = rgb, size=ps)
            else: # variable colour
                rgb = p3s.BufferAttribute( array=rgb )
                geometry = p3s.BufferGeometry( attributes={'position': pts, 'color' : rgb  })
                material = p3s.PointsMaterial(vertexColors = 'VertexColors', size=ps)
        else:
            geometry = p3s.BufferGeometry( attributes={'position': pts })
            material = p3s.PointsMaterial(color='red',  size=ps)
        
        # build point cloud object
        cloud = p3s.Points(
            geometry=geometry,
            material=material
        )

        # link to GUI elements
        ipywidgets.jslink((self._size_slider, 'value'), (material, 'size'))


        # add picker
        if True:
            picker = p3s.Picker(controlling = cloud, all=True, event='dblclick', pointThreshold = 0.01)
            self._renderer.controls = self._renderer.controls + [picker]
            hover_point = p3s.Mesh(geometry=p3s.SphereGeometry(radius=2*ps),
                            material=p3s.MeshBasicMaterial(color='red'))
            self._scene.add(hover_point)
            ipywidgets.jslink((hover_point, 'position'), (picker, 'point'))

            def pick(evt):
                self._picker = picker
                # todo; do something useful here
                
            picker.observe(pick)
            
        point_obj = {"geometry": geometry, 
                     "mesh": cloud, 
                     "material": material,
                     "bounds":bounds,
                     "max":ma,
                     "min":mi,
                     "type": "Points", 
                     "wireframe": None}
        return self.__add_object(name, point_obj)
    
    def clip( self, name, direction=[1,0,0], width=None, visible=False, pos=0 ):
        self._renderer.localClippingEnabled = True;

        # get min and max
        direction = np.array(direction) / np.linalg.norm(direction) # ensure this has a unit length
        mi = np.min( [  o['min'] for o in self._objects.values() ], axis=0 )
        ma = np.max( [  o['max'] for o in self._objects.values() ], axis=0 )
        bounds = self.__get_bbox([mi,ma])
        diag = ma - mi
        length = np.abs( np.dot(diag, direction) ) # travel length
        
        # create geometry
        geometry = p3s.PlaneGeometry( 1, 1 )
        material = p3s.MeshPhongMaterial(color = "red", side="DoubleSide", diffuse="blue", opacity=0.2, transparent=True)
        plane = p3s.Mesh( geometry, material )
        plane.lookAt(direction)
        plane.position = self._centroid
        plane.visible = visible
        #self.plane = plane
        #self._scene.add(plane)
        
        # add slider
        slider = ipywidgets.FloatSlider(value=pos, min=-length/2, max=length/2, description='%s pos'%name)
        self._clip_slider[name] = slider
        wslider = None
        if width is not None:
            wslider = ipywidgets.FloatSlider(value=width, min=0, max=length/2, description='%s width'%name)
            self._clip_slider[name+'_width'] = wslider

        # add object
        clip_obj = {"geometry": geometry, 
                     "mesh": plane, 
                     "material": material,
                     "bounds":bounds,
                     "max":ma,
                     "min":mi,
                     "type": "Clip", 
                     "direction" : direction,
                     "width" : width,
                     "wslider" : wslider,
                     "slider" : slider,
                     "wireframe": None}
        out =  self.__add_object(name, clip_obj)

        # setup interaction
        def update_clip(change):
            clips = []
            for k,o in self._objects.items():
                if o['type'] == 'Clip': 
                    s = o['slider']
                    i = s.value
                    plane = o['mesh']
                    direction = o['direction']
                    width = o['width']
                    if width is not None:
                        width = o['wslider'].value
                    
                    plane.position = [c+o for c,o in zip(self._centroid, i*direction) ]
                    clips.append(p3s.Plane(tuple([v for v in direction]),0.05-np.dot(plane.position, direction)))
                    if width is not None:
                        clips.append(p3s.Plane(tuple([-v for v in direction]),width-np.dot(plane.position, -direction)))
                    #self._renderer.clippingPlanes = [p3s.Plane(tuple([v for v in direction]),0.05-np.dot(plane.position, direction))]
            self._renderer.clippingPlanes = clips # update clipping planes
        update_clip(None)
        slider.observe(update_clip)
        if width is not None:
            wslider.observe(update_clip)
        
        return out
    
    def show(self):
        from ipywidgets import AppLayout
        from ipywidgets import HTML

        left = [ self._size_slider]+list(self._clip_slider.values())
        left += [HTML('<hr/>')]
        for k,v in self._objects.items():
            left.append(
                ipywidgets.Checkbox(
                    value=v['mesh'].visible,
                    description=k,
                    disabled=False,
                    indent=True
                )
            )
            ipywidgets.jslink((left[-1], 'value'), (v['mesh'], 'visible'))

        store = ipywidgets.Button(description="Store point")
        def clk(e):
            if self._picker is not None:
                if self._picker.index is not None:
                    self.pickedPoints.append( self._picker.point )
                    self.pickedIDs.append( self._picker.index )

                    # add mesh
                    ps = 0.1
                    if self._size_slider is not None:
                        ps = self._size_slider.value
                    hover_point = p3s.Mesh(geometry=p3s.SphereGeometry(radius=2*ps),
                                           material=p3s.MeshBasicMaterial(color='yellow'))
                    self._scene.add(hover_point)

        store.on_click(clk)
        right = [store]

        return AppLayout(header=None,
          left_sidebar=ipywidgets.VBox([i for i in left if i is not None]),
          center=self._renderer,
          right_sidebar=ipywidgets.VBox([i for i in right if i is not None]),
          footer=None)