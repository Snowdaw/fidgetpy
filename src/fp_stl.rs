use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray, PyArrayMethods, PyUntypedArrayMethods, PyReadonlyArray2};
use ndarray:: Ix2;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use crate::fp_mesh::PyMesh;

// Define the mesh data enum to handle both NumPy arrays and Python lists
enum MeshData<'py> {
    // Try to match the types used in fidget::mesh::Mesh
    // - vertices: Vec<nalgebra::Vector3<f32>>
    // - triangles: Vec<nalgebra::Vector3<usize>>
    Numpy(PyReadonlyArray2<'py, f32>, PyReadonlyArray2<'py, u64>),
    List(&'py Bound<'py, PyList>, &'py Bound<'py, PyList>),
}

pub fn save_stl(py: Python<'_>, mesh: &PyMesh, filepath: String) -> PyResult<()> {
    // Create output file
    let path = Path::new(&filepath);
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write STL header (80 bytes) - use the same header as Fidget CLI
    const HEADER: &[u8] = b"This is a binary STL file exported by Fidget";
    writer.write_all(HEADER)?;
    // Pad to 80 bytes
    writer.write_all(&[0u8; 80 - HEADER.len()])?;
    
    // Get Bound objects for Pythons objects
    let vertices_obj = mesh.vertices.bind(py);
    let triangles_obj = mesh.triangles.bind(py);

    // Try to interpret the Python objects as NumPy arrays or Lists
    let mesh_data = if let Ok(verts_arr) = vertices_obj.downcast::<PyArray<f32, Ix2>>() {
        if let Ok(tris_arr) = triangles_obj.downcast::<PyArray<u64, Ix2>>() {
            // Successfully downcasted both to NumPy arrays
            let verts_ro = verts_arr.readonly(); // Get readonly view
            let tris_ro = tris_arr.readonly();   // Get readonly view

            // Validate shapes
            if verts_ro.ndim() != 2 || verts_ro.shape()[1] != 3 {
                 return Err(PyValueError::new_err(format!(
                     "NumPy vertices array must have shape (N, 3), found {:?}", verts_ro.shape()
                 )));
            }
            if tris_ro.ndim() != 2 || tris_ro.shape()[1] != 3 {
                 return Err(PyValueError::new_err(format!(
                     "NumPy triangles array must have shape (M, 3) and dtype u64, found {:?} and dtype {}",
                     tris_ro.shape(), tris_arr.dtype().str()?
                 )));
            }
            Ok(MeshData::Numpy(verts_ro, tris_ro))
        } else {
             // Vertices are NumPy, but triangles are not (or not u64)
             Err(PyValueError::new_err("Vertices are NumPy array, but Triangles are not a NumPy array of u64. Both must be compatible (both NumPy arrays or both Lists)."))
        }
    } else if let Ok(verts_list) = vertices_obj.downcast::<PyList>() {
         if let Ok(tris_list) = triangles_obj.downcast::<PyList>() {
             // Both are Python lists
             Ok(MeshData::List(verts_list, tris_list))
         } else {
             // Vertices are List, but triangles are not
             Err(PyValueError::new_err("Vertices are List, but Triangles are not. Both must be compatible (both NumPy arrays or both Lists)."))
         }
    } else {
         // Vertices object is neither a recognizable NumPy array nor a List
         Err(PyValueError::new_err("Vertices object could not be interpreted as a NumPy array or a List."))
    };

    // Handle potential errors from the checks above
    let mesh_data = mesh_data?;

    // Determine the number of triangles based on the data type
    let num_triangles = match &mesh_data {
        MeshData::Numpy(_, tris_arr) => tris_arr.shape()[0],
        MeshData::List(_, tris_list) => tris_list.len(),
    } as u32;

    // Write number of triangles (4 bytes)
    writer.write_all(&num_triangles.to_le_bytes())?;
    
    // For each triangle, write the normal and vertices
    for i in 0..num_triangles as usize {
        // Extract triangle indices based on data type
        let (idx1, idx2, idx3) = match &mesh_data {
            MeshData::Numpy(_, tris_arr) => {
                // Prefer safe indexing with checked bounds via .as_array()
                let tris_view = tris_arr.as_array();
                // Check bounds explicitly just in case (although shape check *should* cover this)
                if i >= tris_view.shape()[0] {
                    return Err(PyValueError::new_err(format!("Triangle index {} out of bounds", i)));
                }
                // Indices are u64, cast to usize for indexing vertices
                (
                    tris_view[[i, 0]] as usize,
                    tris_view[[i, 1]] as usize,
                    tris_view[[i, 2]] as usize,
                )
            },
            MeshData::List(_, tris_list) => {
                // Existing list logic (seems correct)
                let triangle = tris_list.get_item(i)?;
                let indices = triangle.downcast::<PyList>()?;
                if indices.len() != 3 {
                    return Err(PyValueError::new_err("Each triangle in the list must have exactly 3 indices"));
                }
                (
                    // Ensure indices are extracted as usize if that's how vertices will be indexed
                    indices.get_item(0)?.extract::<usize>()?,
                    indices.get_item(1)?.extract::<usize>()?,
                    indices.get_item(2)?.extract::<usize>()?
                )
            }
        };
        
        // Get vertices
        let v1 = extract_vertex(&mesh_data, idx1)?;
        let v2 = extract_vertex(&mesh_data, idx2)?;
        let v3 = extract_vertex(&mesh_data, idx3)?;
        
        // Calculate normal vector using cross product - similar to Fidget CLI
        // Not the _best_ way to calculate a normal, but good enough (comment from Fidget CLI)
        let ab = (v2.0 - v1.0, v2.1 - v1.1, v2.2 - v1.2);
        let ac = (v3.0 - v1.0, v3.1 - v1.1, v3.2 - v1.2);
        
        // Cross product (ab × ac)
        let normal = (
            ab.1 * ac.2 - ab.2 * ac.1,
            ab.2 * ac.0 - ab.0 * ac.2,
            ab.0 * ac.1 - ab.1 * ac.0
        );
        
        // Write normal components
        for p in &[normal.0, normal.1, normal.2] {
            writer.write_all(&p.to_le_bytes())?;
        }
        
        // Write vertices - similar to Fidget CLI
        for v in &[v1, v2, v3] {
            for p in &[v.0, v.1, v.2] {
                writer.write_all(&p.to_le_bytes())?;
            }
        }
        
        // Write attribute byte count (2 bytes) - typically zero
        writer.write_all(&[0u8, 0u8])?;
    }
    
    Ok(())
}

// Helper function to extract vertex coordinates
fn extract_vertex(mesh_data: &MeshData, idx: usize) -> PyResult<(f32, f32, f32)> {
    match mesh_data {
        MeshData::Numpy(verts_arr, _) => {
            // Access the readonly NumPy array view
            let verts_view = verts_arr.as_array();
            // Check bounds explicitly for safety
            if idx >= verts_view.shape()[0] || verts_view.shape()[1] < 3 {
                 return Err(PyValueError::new_err(format!("Vertex index {} out of bounds or vertex data format incorrect", idx)));
            }
            // Use safe indexing
            Ok((
                verts_view[[idx, 0]],
                verts_view[[idx, 1]],
                verts_view[[idx, 2]],
            ))
        },
        MeshData::List(verts_list, _) => {
            // Existing list logic (seems correct)
            let vertex = verts_list.get_item(idx)?;
            let coords = vertex.downcast::<PyList>()?;
            if coords.len() != 3 {
                return Err(PyValueError::new_err("Each vertex in the list must have exactly 3 coordinates"));
            }
            Ok((
                coords.get_item(0)?.extract::<f32>()?,
                coords.get_item(1)?.extract::<f32>()?,
                coords.get_item(2)?.extract::<f32>()?
            ))
        }
    }
}

