/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "raw/RawModel.hpp"

#include "FbxBlendShapesAccess.hpp"
#include "FbxLayerElementAccess.hpp"
#include "FbxSkinningAccess.hpp"
#include "materials/RoughnessMetallicMaterials.hpp"
#include "materials/TraditionalMaterials.hpp"

struct Polygon {
  Polygon(int fbxPolyIndex)
      : fbxPolyIndex(fbxPolyIndex), hasTranslucentVertices(false), rawPolyIndex(-1) {}
  const int fbxPolyIndex;
  bool hasTranslucentVertices;
  int rawPolyIndex;
};

struct Triangle {
  Triangle(Polygon& polygon, int v1, int v2, int v3)
      : polygon(polygon), vertexIndices{v1, v2, v3}, rawVertexIndices{-1, -1, -1} {}
  Polygon& polygon;
  const int vertexIndices[3];
  int rawVertexIndices[3];
};

struct Triangulation {
  Triangulation(FbxMesh* mesh) : mesh(mesh) {}
  FbxMesh* const mesh;
  std::vector<Polygon> polygons;
  std::vector<Triangle> triangles;
};

struct FbxMeshAccess {
  FbxMesh* mesh;
  const FbxLayerElementAccess<FbxVector4> normalLayer;
  const FbxLayerElementAccess<FbxVector4> binormalLayer;
  const FbxLayerElementAccess<FbxVector4> tangentLayer;
  const FbxLayerElementAccess<FbxColor> colorLayer;
  const FbxLayerElementAccess<FbxVector2> uvLayer0;
  const FbxLayerElementAccess<FbxVector2> uvLayer1;
  const FbxSkinningAccess skinning;
  const FbxMaterialsAccess materials;
  const FbxBlendShapesAccess blendShapes;

  FbxMeshAccess(
      FbxScene* scene,
      FbxNode* node,
      const std::map<const FbxTexture*, FbxString>& textureLocations);
};

enum FbxMeshTriangulationMethod {
  FBX_SDK,
  FB_NGON_ENCODING,
};

class FbxMesh2Raw {
 public:
  FbxMesh2Raw(
      RawModel& raw,
      FbxScene* scene,
      FbxNode* node,
      FbxMeshTriangulationMethod triangulationMethod,
      const std::map<const FbxTexture*, FbxString>& textureLocations);

  void Process();

 private:
  RawSurface& ProcessSurface(Triangulation& triangulation, const FbxMeshAccess& meshAccess);
  void ProcessTriangles(
      Triangulation& triangulation,
      RawSurface& surface,
      const FbxMeshAccess& meshAccess);
  void ProcessPolygons(
      Triangulation& triangulation,
      RawSurface& surface,
      const FbxMeshAccess& meshAccess);
  std::unique_ptr<Triangulation> TriangulateUsingSDK();
  std::unique_ptr<Triangulation> TriangulateForNgonEncoding();

  RawModel& raw;
  FbxScene* scene;
  FbxNode* node;
  const FbxMeshTriangulationMethod triangulationMethod;
  const std::map<const FbxTexture*, FbxString>& textureLocations;
};
