
var<private> rand_pcg4d: vec4<u32>;

fn init_pcg4d(v: vec4<u32>)
{
    rand_pcg4d = v * 1664525u + 1013904223u;

    rand_pcg4d.x = rand_pcg4d.x + (rand_pcg4d.y * rand_pcg4d.w);
    rand_pcg4d.y = rand_pcg4d.y + (rand_pcg4d.z * rand_pcg4d.x);
    rand_pcg4d.z = rand_pcg4d.z + (rand_pcg4d.x * rand_pcg4d.y);
    rand_pcg4d.w = rand_pcg4d.w + (rand_pcg4d.y * rand_pcg4d.z);

    let _rand_pcg4d = vec4<u32>(rand_pcg4d.x >> 16u,rand_pcg4d.y >> 16u,rand_pcg4d.z >> 16u,rand_pcg4d.w >> 16u);
    rand_pcg4d = rand_pcg4d ^ _rand_pcg4d;

    // rand_pcg4d = rand_pcg4d ^ (rand_pcg4d >> 16u);
    rand_pcg4d.x = rand_pcg4d.x + (rand_pcg4d.y * rand_pcg4d.w);
    rand_pcg4d.y = rand_pcg4d.y + (rand_pcg4d.z * rand_pcg4d.x);
    rand_pcg4d.z = rand_pcg4d.z + (rand_pcg4d.x * rand_pcg4d.y);
    rand_pcg4d.w = rand_pcg4d.w + (rand_pcg4d.y * rand_pcg4d.z);
}

fn f32_zero_to_one(seed: u32) -> f32 {
    return bitcast<f32>(0x3f800000u | (seed & 0x007fffffu)) - 1.0;
}

fn f32_negone_to_one(seed: u32) -> f32 {
    return bitcast<f32>(0x40000000u | (seed & 0x007fffffu)) - 3.0;
}

fn rescale(value: f32, min: f32, max: f32) -> f32 {
    return (value * (max - min)) + min;
}

fn random_in_unit_sphere() -> vec3<f32> {
    let phi = 2.0 * PI * f32_zero_to_one(rand_pcg4d.x);
    let cos_theta = 2.0 * f32_zero_to_one(rand_pcg4d.y) - 1.0;
    let u = f32_zero_to_one(rand_pcg4d.z);

    let theta = acos(cos_theta);
    let r = pow(u, 1.0 / 3.0);

    let x = r * sin(theta) * cos(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(theta);
    return vec3<f32>(x,y,z);
}

fn random_in_cube() -> vec3<f32> {
    let x = f32_negone_to_one(rand_pcg4d.x);
    let y = f32_negone_to_one(rand_pcg4d.y);
    let z = f32_negone_to_one(rand_pcg4d.z);
    return vec3<f32>(x,y,z);
}



// Probably not need this, just point in cube is fine
// from this: https://stackoverflow.com/questions/18182376/figuring-out-how-much-of-the-side-of-a-cube-is-visible
// TODO: check if the dot products of 3 negative face indeed add to -1

// var<private> FACE_NORMALS: array<vec3<f32>,6> = 
//     array<vec3<f32>,6>(  
//         vec3<f32>(-1f,0f,0f),vec3<f32>(0f,-1f,0f),vec3<f32>(0f,0f,-1f),
//         vec3<f32>(1f,0f,0f), vec3<f32>(0f,1f,0f), vec3<f32>(0f,0f,1f)
//                                     );

// fn random_to_cube_face(view_vector: vec3<f32>) -> vec3<f32> {
//     let a1 = rescale(u32_to_f32(rand_pcg4d.x), -1f, 1f);
//     let a2 = rescale(u32_to_f32(rand_pcg4d.y), -1f, 1f);

//     let choice_idx = u32_to_f32(rand_pcg4d.z);

//     var faces = array<vec3<f32>,3>(vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0));
//     var thres = array<f32,3>(0f,0f,0f);

//     let start = rand_pcg4d.z % 6u;
//     var j = 0u;
//     for (var i = start; i < start + 6u; i = i+1u) {
//         let idx = i % 6u;
//         var r = FACE_NORMALS[idx];
//         let d = dot(r,normalize(view_vector));
//         if (d < 0f) {
//             // r[idx % 3u] = -r[idx % 3u];
//             r[(idx + 1u) % 3u] = a1;
//             r[(idx + 2u) % 3u] = a2;

//             faces[j] = r;
//             thres[j] = -d;

//             if (j == 2u) {
//                 break;
//             }

//             j = j + 1u;

//             // return r;
//         }
//     }

//     var tot = 0f;
//     for (var i = 0u; i < j; i = i+1u) {
//         tot = tot + thres[i];
//         if (choice_idx < tot) {
//             return faces[i];
//         }
//     }

//     return faces[j];
// }

fn random_cosine_direction() -> vec3<f32> {
    let r1 = f32_zero_to_one(rand_pcg4d.x);
    let r2 = f32_zero_to_one(rand_pcg4d.y);
    let z = sqrt(1.0-r2);

    let phi = 2.0*PI*r1;
    let x = cos(phi)*sqrt(r2);
    let y = sin(phi)*sqrt(r2);

    return vec3<f32>(x, y, z);
}

// TODO: update so it can handle non-spherical shapes
fn random_to_sphere(radius: f32, distance_squared: f32) -> vec3<f32> {
    let r1 = f32_zero_to_one(rand_pcg4d.x);
    let r2 = f32_zero_to_one(rand_pcg4d.y);
    let z = 1.0 + r2*(sqrt(1.0-radius*radius/distance_squared) - 1.0);

    let phi = 2.0*PI*r1;
    let scale = sqrt(1.0-(z*z));
    let x = cos(phi)*scale;
    let y = sin(phi)*scale;

    return vec3<f32>(x, y, z);
}

fn random_uniform_direction() -> vec3<f32> {
    let r1 = f32_zero_to_one(rand_pcg4d.x);
    let r2 = f32_zero_to_one(rand_pcg4d.y);
    let x = cos(2.0*PI*r1)*2.0*sqrt(r2*(1.0-r2));
    let y = sin(2.0*PI*r1)*2.0*sqrt(r2*(1.0-r2));
    let z = 1.0 - 2.0*r2;

    return vec3<f32>(x, y, z);
}

fn random_uniform_on_hemisphere() -> vec3<f32> {
    let azimuthal = 2.0*PI*f32_zero_to_one(rand_pcg4d.x);
    let z = f32_zero_to_one(rand_pcg4d.y);

    let xyproj = sqrt(1.0-(z*z));

    let x = cos(azimuthal)*xyproj;
    let y = sin(azimuthal)*xyproj;

    return vec3<f32>(x, y, z);
}

// fn random_uniform_direction(radius: f32) -> vec3<f32>
// {
//     let theta: f32 = rescale(f32_zero_to_one(rand_pcg4d.x), 0.0, PI * 2.0);
//     let phi: f32 = acos(rescale(f32_zero_to_one(rand_pcg4d.y), -1.0, 1.0));

//     let x: f32 = sin(phi) * cos(theta);
//     let y: f32 = sin(phi) * sin(theta);
//     let z: f32 = cos(phi);

//     return normalize(vec3<f32>(x,y,z) * radius);
// }

fn hemisphericalRand(normal: vec3<f32>) -> vec3<f32>
{
    let in_unit_sphere = random_uniform_direction();
    if (dot(in_unit_sphere, normal) > 0.0) { // In the same hemisphere as the normal
        return in_unit_sphere;
    }
    return -in_unit_sphere;
}

fn random_light() -> ObjectParam {
    let i = u32(rescale(f32_zero_to_one(rand_pcg4d.x), f32(ubo.lights_offset), f32(ubo.n_objects)));
    return object_params.ObjectParams[i];
}

fn random_to_light(light: ObjectParam, origin: vec3<f32>) -> vec3<f32> {
    let center = (light.transform * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;
    let direction = center - origin;
    let distance_squared = pow(length(direction),2.0);

    let onb = onb(normalize(direction));

    let scale = vec3<f32>(length(light.transform[0].xyz),length(light.transform[1].xyz),length(light.transform[2].xyz));

    // TODO: fix the negative offset - lots of hot pixels without it, something wrong
    let radius = max(max(scale.x,scale.y),scale.z) - 0.05f;
    
    if (light.model_type == 0u) {
        let r = onb * random_to_sphere(radius,distance_squared);
        return r;
    }
    else if (light.model_type == 1u) {
        return vec3<f32>(0.0);
    }
    else if (light.model_type == 2u) {
        let p = random_in_cube();
        // let p = random_to_cube_face((light.inverse_transform * vec4<f32>(direction,0f)).xyz);
        let r = (light.transform * vec4<f32>(p,1f)).xyz;
        return normalize(r - origin);
    }

    // TODO: for model mesh, choose random triangle, then random point on it
    return vec3<f32>(1.0);
}