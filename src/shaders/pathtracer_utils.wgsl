#define_import_path rengin::pathtracer_utils

#import rengin::types::ObjectParam
#import rengin::constants::PI

fn onb(n: vec3<f32>) -> mat3x3<f32> {
    var out: mat3x3<f32>;
    out[2] = n;

    var a = vec3<f32>(1.0, 0.0, 0.0);
    if abs(out[2].x) > 0.9 {
        a = vec3<f32>(0.0, 1.0, 0.0);
    }
    out[1] = normalize(cross(out[2], a));
    out[0] = cross(out[2], out[1]);

    return out;
}

fn _onb(n: vec3<f32>) -> mat3x3<f32> {
    var out: mat3x3<f32>;

    out[2] = n;
    if n.z < -0.99995 {
        out[0] = vec3<f32>(0.0, -1.0, 0.0);
        out[1] = vec3<f32>(-1.0, 0.0, 0.0);
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        out[0] = vec3<f32>(1.0 - n.x * n.x * a, b, -n.x);
        out[1] = vec3<f32>(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return out;
}


fn onb_local(v: vec3<f32>, onb: mat3x3<f32>) -> vec3<f32> {
    // return v.x*onb[0] + v.y*onb[1] + v.z*onb[2];
    return onb * v;
}


// TODO: check the initial size cube and sphere and centroid coordinates and adjust accordingly
fn surface_area(object: ObjectParam) -> f32 {
    let scale = vec3<f32>(length(object.transform[0].xyz), length(object.transform[1].xyz), length(object.transform[2].xyz));
    if object.model_type == 0u {
        return 4.0 * PI * pow((pow(scale.x * scale.z, 1.6075) + pow(scale.z * scale.y, 1.6075) + pow(scale.x * scale.y, 1.6075)) / 3.0, 1.0 / 1.6075);
    } else if object.model_type == 1u {
        // TODO
        return 1.0;
    } else if object.model_type == 2u {
        // return (343.0 - 213.0) * (332.0 - 227.0);
        return 2.0 * ((2.0 * scale.x * scale.z) + (2.0 * scale.z * scale.y) + (2.0 * scale.x * scale.y));
    } else if object.model_type == 9u {
        return 1.0;
    }

    // TODO
    return 1.0;
}
