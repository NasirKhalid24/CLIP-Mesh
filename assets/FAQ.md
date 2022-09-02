# Tips, tricks and FAQ

## Do I need a powerful GPU?

Not really, this code was written to work on typical gaming GPUs, if you are having memory issues try reducing the following parameters (listed in order of memory consumption):

- train_res
- batch_size
- texture_resolution

If you are having issues with the first step of the program where it cannot load and use the diffusion prior try removing the prior all together by setting the ```prior_path``` in your config file to an empty value while also reducing the values

Note that doing all this may reduce quality of results

## Prompt Engineering 

Prompt engineering was not explored at all, so there's not much I can share here - if you do find prompts that improve results please submit a pull request and them here

## The texture is quite noisy 

Try increasing ```train_res``` or set the range of ```dist_min``` and ```dist_max``` to a lower value - additionally you could also reduce ```texture_resolution``` and increase blur parameters

## The generated shape is flat on some sides

Try increasing the ```batch_size```, increase the range of ```dist_min``` and ```dist_max``` and ensure that all ```aug_``` parameters are set to true

## I added a custom mesh to generate with and the program just crashes

This could be three reasons that I know of:

- The mesh is non-manifold is which case the limit subdivision does not work, try to remesh the shape
- There is a vertice in the mesh whose valence is outside the range of [3, 50] - hence the limit subdivision does not work
- There are a huge number of vertices, the spheres used in the papaer have about 600 vertices

## How can I setup a custom scene to generate in

I recommend setting up your scene in Blender and importing ```primitives/sphere.obj``` in to the scene - repositioning, resizing it as required. Then save the newly positoned/sized sphere as a new .obj file and save the rest of your scene meshes as .obj files (ensure the textures are baked and UV unwrapped)

To generate with this scene your config file would have all the parameters as is (may need to change camera params) and then towards the end your meshes parameters would like as follows

```yaml
....


# Mesh Parameters

## Add meshes to the scene here
meshes: 
  - path/to/exported/sphere.obj
  - path/to/exported/scene_mesh_1.obj
  - path/to/exported/scene_mesh_2.obj
  ....

## Unit scale the meshes? No need as it was done in Blender
unit: 
  - false
  - false
  - false
  ....

## Sphere is first and it will be optimized, the rest are just constant scene objects
train_mesh_idx:
  - [verts, texture, normal, true]
  - []
  - []
  ....

## No scaling as it was done in Blender
scales:
  - 1.0
  - 1.0
  - 1.0
  ....

## No positioning as it was done in Blender
offsets:
  - [0.0,  0.0,  0.0]
  - [0.0,  0.0,  0.0]
  - [0.0,  0.0,  0.0]
  ....

```

You could retexture a mesh in your scene by just setting its corresponding ```train_mesh_idx``` to ```[texture, normal, true]``` 

## I cannot generate ____ the results are not good as ____

Text to 3D is much newer than Text to Image generation and therefore the results are not up to par - additionally while most text to image models rely on a huge set of Image-Text data there is no 3D-Text data available at the same scale

Complex 3D objects with massive variation are tough to generate as CLIP embeddings are limited in the information they hold - it is recommended to generate a single object at a time even something like a car is tough to generate do but a tyre is not

You could also try using other text to 3D generation/stylization techniques such as [Dreamfields](https://ajayj.com/dreamfields), [CLIPForge](https://github.com/AutodeskAILab/Clip-Forge), [Text2Mesh](https://github.com/threedle/text2mesh)