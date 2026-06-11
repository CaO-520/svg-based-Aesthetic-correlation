import base64
import html
import numpy as np
from svgpathtools import (
    Path, Arc, CubicBezier, QuadraticBezier,
    svgstr2paths)
import os 
import re
import matplotlib.colors as mcolors
import yaml
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

try:
    from noise import pnoise1
except ImportError:
    def pnoise1(value, octaves=1):
        noise = 0.0
        amplitude = 1.0
        frequency = 1.0
        amplitude_sum = 0.0
        for _ in range(octaves):
            noise += np.sin(value * frequency * 12.9898) * amplitude
            amplitude_sum += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        return noise / amplitude_sum if amplitude_sum else 0.0


def rasterize_svg(svg, output_width=512, output_height=512, background_color='white'):
    svg_data_url = (
        'data:image/svg+xml;base64,'
        + base64.b64encode(svg.encode('utf-8')).decode('ascii')
    )
    background = html.escape(background_color)

    page_html = f'''
    <!doctype html>
    <html>
      <head>
        <style>
          html, body {{
            margin: 0;
            width: {output_width}px;
            height: {output_height}px;
            overflow: hidden;
            background: {background};
          }}
          #frame {{
            align-items: center;
            background: {background};
            display: flex;
            height: {output_height}px;
            justify-content: center;
            width: {output_width}px;
          }}
          #frame > img {{
            display: block;
            height: 100%;
            max-height: 100%;
            max-width: 100%;
            object-fit: contain;
            width: 100%;
          }}
        </style>
      </head>
      <body>
        <div id="frame">
          <img src="{svg_data_url}" />
        </div>
      </body>
    </html>
    '''

    with sync_playwright() as playwright:
        browser = None
        last_error = None
        for channel in ('msedge', 'chrome', None):
            try:
                browser = playwright.chromium.launch(channel=channel, headless=True)
                break
            except Exception as error:
                last_error = error

        if browser is None:
            raise RuntimeError(
                'Could not launch a browser. Install Microsoft Edge/Chrome, '
                'or run: python -m playwright install chromium'
            ) from last_error

        try:
            page = browser.new_page(
                viewport={'width': output_width, 'height': output_height}
            )
            page.set_content(page_html, wait_until='load')
            page.wait_for_function(
                '''
                () => {
                  const image = document.querySelector('img');
                  return image && image.complete && image.naturalWidth > 0;
                }
                '''
            )
            return page.locator('#frame').screenshot(type='png', scale='css')
        finally:
            browser.close()


def range_to_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return {'from': value[0], 'to': value[1]}
    return {'from': value, 'to': value}


class SVGTransforms:
    def __init__(self, transformations):
        self.transformations = transformations
        self.noise_std = self.transformations.get('noise_std', False) 
        self.noise_type = self.transformations.get('noise_type', False)
        self.rotate = self.transformations.get('rotate', False)
        self.shift_re = self.transformations.get('shift_re', False)
        self.shift_im = self.transformations.get('shift_im', False)
        self.scale = self.transformations.get('scale', False)
        self.color_noise = self.transformations.get('color_noise', False)
        self.p = self.transformations.get('p', 0.5)
        self.color_change = self.transformations.get('color_change', False)
        self.colors = self.transformations.get('colors', ['#ff0000', '#0000ff', '#000000'])
        self.element_drop_rate = self.transformations.get('element_drop_rate', False)
        self.element_drop_prob = self.transformations.get('element_drop_prob', False)
        self.element_drop_count = self.transformations.get('element_drop_count', False)
        self.scale_factor = 1.0
        self.sampled_element_drop_rate = 0
        self.sampled_element_drop_count = 0

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return cls.from_degradation_config(config.get('degradation', {}))

    @classmethod
    def from_degradation_config(cls, degradation_config):
        random_seed = degradation_config.get('random_seed')
        if random_seed is not None:
            np.random.seed(random_seed)

        methods = set(degradation_config.get('methods', []))
        params = degradation_config.get('parameters', {})
        transformations = {}

        if 'rotate' in methods:
            transformations['rotate'] = range_to_dict(params.get('rotate_deg', [-8, 8]))

        if 'shift' in methods:
            shift_px = range_to_dict(params.get('shift_px', [-6, 6]))
            transformations['shift_re'] = range_to_dict(params.get('shift_re_px', shift_px))
            transformations['shift_im'] = range_to_dict(params.get('shift_im_px', shift_px))

        if 'scale' in methods:
            transformations['scale'] = range_to_dict(params.get('scale_factor', [0.9, 1.1]))

        if 'path_noise' in methods:
            transformations['noise_std'] = range_to_dict(params.get('path_noise_std', [0.2, 1.0]))
            transformations['noise_type'] = params.get('path_noise_type', 'gaussian')

        if 'color_noise' in methods:
            transformations['color_noise'] = range_to_dict(params.get('color_noise_std', [0.02, 0.08]))

        if 'color_change' in methods:
            transformations['color_change'] = True
            transformations['colors'] = params.get('colors', ['#ff0000', '#0000ff', '#000000'])

        if 'element_drop' in methods:
            if 'element_drop_rate' in params:
                transformations['element_drop_rate'] = range_to_dict(params['element_drop_rate'])
            if 'element_drop_count' in params:
                transformations['element_drop_count'] = range_to_dict(params['element_drop_count'])
            if 'element_drop_prob' in params:
                transformations['element_drop_prob'] = range_to_dict(params['element_drop_prob'])

        return cls(transformations)

    def sample_transformations(self):
        if self.rotate:
            a, b = self.rotate['from'], self.rotate['to']
            rotation_angle = np.random.uniform(a, b)
            self.rotation_angle = rotation_angle

        if self.shift_re:
            self.shift_real = np.random.uniform(self.shift_re['from'], self.shift_re['to'])
        else:
            self.shift_real = 0

        if self.shift_im:
            self.shift_imag = np.random.uniform(self.shift_im['from'], self.shift_im['to'])
        else:
            self.shift_imag = 0

        if self.scale:
            self.scale_factor = np.random.uniform(self.scale['from'], self.scale['to'])

        if self.color_noise:
            self.color_noise_std = np.random.uniform(self.color_noise['from'], self.color_noise['to'])

        if self.element_drop_prob:
            self.sampled_element_drop_prob = np.random.uniform(
                self.element_drop_prob['from'],
                self.element_drop_prob['to'],
            )
        else:
            self.sampled_element_drop_prob = 0

        if self.element_drop_rate:
            self.sampled_element_drop_rate = np.random.uniform(
                self.element_drop_rate['from'],
                self.element_drop_rate['to'],
            )
        else:
            self.sampled_element_drop_rate = 0

        if self.element_drop_count:
            drop_count = np.random.uniform(
                self.element_drop_count['from'],
                self.element_drop_count['to'],
            )
            self.sampled_element_drop_count = int(round(drop_count))
        else:
            self.sampled_element_drop_count = 0


    def paths2str(self, groupped_paths, svg_opening_tag='<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'):
        
        keys_to_exclude = ['d', 'cx', 'cy', 'rx', 'ry']
        all_groups_srt = ''
        for group, elements in groupped_paths.items():
            group_attributes, paths_and_attributes = elements.get('attrs', {}), elements.get('paths', [])
            group_attr_str = ' '.join(f'{key}="{value}"' for key, value in group_attributes.items())
            path_strings = []
            path_str = ''
            for path, attributes in paths_and_attributes:
                path_attr_str = ''
                d_str = path.d()
                
                for key, value in attributes.items():
                    if key not in keys_to_exclude:
                        path_attr_str += f' {key}="{value}"'

                path_strings.append(f'<path d="{d_str}"{path_attr_str} />')
            path_str = "\n".join(path_strings)
            if 'no_group'in group:
                group_str = path_str
            else:
                group_str = f'<g {group_attr_str}>\n{path_str}\n</g>\n'
            all_groups_srt += group_str
        svg = f'{svg_opening_tag}\n{all_groups_srt}</svg>'
        return svg
    
    def add_noise(self, seg):        
        noise_scale = np.random.uniform(self.noise_std['from'], self.noise_std['to'])
        if self.noise_type == 'gaussian':
            noise_sample = np.random.normal(loc=0.0, scale=noise_scale) + \
                        1j * np.random.normal(loc=0.0, scale=noise_scale)
        elif self.noise_type == 'perlin':
            noise_sample = complex(pnoise1(np.random.random(), octaves=2), pnoise1(np.random.random(), octaves=2))*noise_scale

        if isinstance(seg, CubicBezier):
            seg.control1 = seg.control1 + noise_sample
            seg.control2 = seg.control2 + noise_sample
        elif isinstance(seg, QuadraticBezier):
            seg.control = seg.control + noise_sample
        elif isinstance(seg, Arc):
            seg.radius = seg.radius + noise_sample

                
        return seg
    
    def do_rotate(self, path, viewbox_width, viewbox_height):
        if self.rotate:
            new_path = path.rotated(self.rotation_angle, complex(viewbox_width/2, viewbox_height/2))
            return new_path
        else:
            return path

    def do_shift(self, path):
        if self.shift_re or self.shift_im:
            return path.translated(complex(self.shift_real, self.shift_imag))
        else:
            return path

    def do_scale(self, path):
        if self.scale:
            return path.scaled(self.scale_factor)
        else:
            return path

    def should_drop_element(self, path_index, drop_indexes):
        if path_index in drop_indexes:
            return True

        if self.sampled_element_drop_prob > 0:
            return np.random.random() < self.sampled_element_drop_prob

        return False
    
    def add_color_noise(self, source_color):
         # Convert color to RGB 
        if source_color.startswith("#"):
            base_color = mcolors.hex2color(source_color)
        else:
            base_color = mcolors.hex2color(mcolors.CSS4_COLORS.get(source_color, '#FFFFFF'))

        # Add noise to each RGB component
        noise = np.random.normal(0, self.color_noise_std, 3)
        noisy_color = np.clip(np.array(base_color) + noise, 0, 1)

        # Convert the RGB color back to hex
        hex_color = mcolors.rgb2hex(noisy_color)

        return hex_color

    def do_color_change(self, attr):
        if 'fill' in attr:
            if self.color_noise or self.color_change:
                fill_value = attr['fill']    
                if fill_value == 'none':
                    new_fill_value = 'none'
                else:
                    if self.color_noise:
                        new_fill_value = self.add_color_noise(fill_value)
                    elif self.color_change:
                        new_fill_value = np.random.choice(self.colors)
                attr['fill'] = new_fill_value
        return attr
    
    def clean_attributes(self, attr):
        attr_out = {}
        if 'fill' in attr:
            attr_out = attr
        elif 'style' in attr:
            fill_values = re.findall('fill:[^;]+', attr['style'])
            if fill_values:
                fill_value = fill_values[0].replace('fill:', '').strip()
                attr_out['fill'] = fill_value
            else:
                attr_out = attr
        else:
            attr_out = attr

        return attr_out

    def get_viewbox_size(self, svg):
        # Try to extract viewBox attribute
        match = re.search(r'viewBox="([^"]+)"', svg)
        if match:
            viewbox = match.group(1)
        else:
            # If viewBox is not found, try to extract width and height attributes
            match = re.search(r'width="([^"]+)px" height="([^"]+)px"', svg)
            if match:
                width, height = match.groups()
                viewbox = f"0 0 {width} {height}"
            else:
                viewbox = "0 0 256 256"  # Default if neither viewBox nor width/height are found
    
        viewbox = [float(x) for x in viewbox.split()]
        viewbox_width, viewbox_height = viewbox[2], viewbox[3]
        return viewbox_width, viewbox_height

    def augment(self, svg, render_image=True):
        if os.path.isfile(svg):
            # open svg file
            with open(svg, 'r') as f:
                svg = f.read()
                
        # Sample transformations for this sample
        self.sample_transformations() 


        # Parse the SVG content
        soup = BeautifulSoup(svg, 'xml')

        # Get opening tag
        svg_opening_tag = re.findall('<svg[^>]+>', svg)[0]

        viewbox_width, viewbox_height = self.get_viewbox_size(svg)

        # Get all svg parents
        groups = soup.findAll()
        
        parsed_groups = []
        path_total = 0
        for i, g in enumerate(groups):
            if g.name == 'g':
                group_id = group_id = g.get('id') if g.get('id') else f'none_{i}'
                group_attrs = g.attrs

            elif g.name == 'svg' or g.name == 'metadata' or g.name == 'defs':
                continue
            
            else:
                group_id = f'no_group_{i}'
                group_attrs = {}
            
            group_svg_string = f'{svg_opening_tag}{str(g)}</svg>'
            try:
                paths, attributes = svgstr2paths(group_svg_string)
            except:
                image = rasterize_svg(svg) if render_image else None
                return svg, image
            if not paths:
                continue

            parsed_groups.append((group_id, group_attrs, paths, attributes))
            path_total += len(paths)

        drop_count = self.sampled_element_drop_count
        if self.sampled_element_drop_rate > 0:
            drop_count = max(drop_count, int(round(path_total * self.sampled_element_drop_rate)))
        if path_total > 1:
            drop_count = min(drop_count, path_total - 1)
        else:
            drop_count = 0
        if drop_count > 0:
            drop_indexes = set(np.random.choice(path_total, drop_count, replace=False))
        else:
            drop_indexes = set()

        # Create the groups of paths based on their original <g> tag
        grouped_paths = {}
        path_index = 0
        for group_id, group_attrs, paths, attributes in parsed_groups:
            paths_and_attributes = []

            # Rotation, shift, scale, noise addition
            for path, attribute in zip(paths, attributes):
                if self.should_drop_element(path_index, drop_indexes):
                    path_index += 1
                    continue

                attr = self.clean_attributes(attribute)
                
                new_path = self.do_rotate(path, viewbox_width, viewbox_height)
                new_path = self.do_shift(new_path)
                new_path = self.do_scale(new_path)
                
                if self.noise_std:
                    # Add noise to path to deform svg
                    noisy_path = []
                    for seg in new_path:
                        noisy_seg = self.add_noise(seg)
                        noisy_path.append(noisy_seg)
                    new_path = Path(*noisy_path)

                # Color change
                attr = self.do_color_change(attr)
                paths_and_attributes.append((new_path, attr))
                path_index += 1

            if not paths_and_attributes:
                continue
            
            grouped_paths[group_id] = {
                'paths': paths_and_attributes, 
                'attrs': group_attrs
                }

        svg = self.paths2str(grouped_paths, svg_opening_tag)
        image = rasterize_svg(svg) if render_image else None

        return svg, image
