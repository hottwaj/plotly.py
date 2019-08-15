from datetime import datetime, date
import numpy
import pandas
import copy
import uuid
from past.builtins import basestring    # pip install future

from functools import partial, reduce

#dependency on plotly package added - actually this is only to load the plotly js script...
from .offline import iplot, init_notebook_mode
init_notebook_mode(connected=False) 

from IPython.core.display import HTML, display

display(HTML("""
<style>
/* for chart subplots tables produced by 'draw_charts_table' */
.table-no-border {
    border: none !important;
}
</style>"""))

import plotly.io as pio
pio.renderers.default = 'iframe'  #required to return a 'text/html' iframe bundle that can then be dropped as html
    
DEFAULT_COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

from itertools import zip_longest

def grouped(iterable, n):
    "group a sequence of objects into a sequence of tuples each containing 'n' objects"
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    #from http://stackoverflow.com/a/5389547/1280629
    return zip_longest(*[iter(iterable)]*n)
    
def charts_table(charts, cols):
    "draw a sequence of HTML charts (e.g. plotly interactive charts) as 'subplots' in a table with 'cols' columns"
    table_content = '<table class="table-no-border">'
    div_ids = []
    for row in grouped(charts, cols):
        table_content += '<tr class="table-no-border">'
        for chart in row:
            if chart is not None:
                #odd re-writing of width and height needed to ensure they are not
                #overwritten by multiple charts plotted simultaneously
                if 'layout' in chart.data_layout:
                    layout = chart.data_layout['layout']
                    layout['width'] = chart.width
                    layout['height'] = chart.height
                table_content += '<td class="table-no-border">%s</td>' \
                                 % chart._repr_mimebundle_()['text/html']
        table_content += '</tr>'
    table_content += '</table>'

    display(HTML(table_content))
    
def percent_axis(axis_settings = {}, tick_precision = 0, hover_precision = 2):
    return dict(axis_settings, **{
        'tickformat': ',.%d%%' % tick_precision,
        'hoverformat': ',.%d%%' % hover_precision, 
    })
    
default_layout = {
    #'yaxis': {
    #    'hoverformat': '.2f', 
    #},
    #'xaxis': {
    #    'hoverformat': '.2f', 
    #},
    'template': 'plotly_white',
    'margin': {
      'l': 60,
      'r': 50,
      'b': 50,
      't': 50,
      'pad': 4
    },
    'autosize': True
}

default_config = {'showLink': False}

def dict_merge(a, b, path=None):
    "merges b into a, recursively copying any subdicts"
    if path is None: path = []
    for key, bval in b.items():
        if key in a and isinstance(a[key], dict) and isinstance(bval, dict):
            dict_merge(a[key], bval, path + [str(key)])
        elif isinstance(bval, dict):
            a[key] = copy.deepcopy(bval)
        else:
            a[key] = bval
    return a

class _PlotlyChartBundle(object):
    """Class for returning a displayable object wrapping a plotly chart.
    This is used to wrap a plotted chart so we can then drop it into a table if required."""
    
    def __init__(self, data_layout, width, height, config):
        self.data_layout = data_layout
        self.width = width
        self.height = height
        self.config = config
        
    def _repr_mimebundle_(self, *args, **kwargs):
        #use iplot to return a renderable bundle
        bundle = iplot(self.data_layout, 
              image_width = self.width, 
              image_height = self.height, 
              config = self.config, 
              return_bundle = True)

        return bundle
        
def scatter(df, x_col, y_col, 
                     groups_col = None, tooltip_cols = [], group_order = None, 
                     layout = dict(), series_dict = dict(), x_order = [], group_colours = dict(),
                     color_col = None, size_col = None,
                     scatter_type = 'scatter',  #could be changed to e.g. scattergl
                     width = 600, height = 400):
    pl_data = []
    if groups_col is not None and color_col is not None:
        raise RuntimeError('Only one of "groups_col" or "color_col" should be provided when calling this function')
        
    if groups_col is not None:
        groups_available = set(df[groups_col])
        sorted_groups = group_order if group_order is not None else sorted(groups_available)
    layout = reduce(dict_merge, [{}, default_layout, layout])  #overwrite default_layout with given layout
    
    if isinstance(x_col, basestring):
        xvals = df[x_col]
    else:
        xvals = x_col
    if isinstance(y_col, basestring):
        yvals = df[y_col]
    else:
        yvals = y_col
                                            
    layout['width'] = width
    layout['height'] = height
        
    def _process_group(grp, grp_vals):
        line_dict = {
          'x': xvals[grp_vals].values,
          'y': yvals[grp_vals].values,
          'mode': 'markers',
          'type': scatter_type,
          'name': grp,
          'marker': { 'size': 7 }
        }
        if tooltip_cols:
            line_dict['text'] = ['<br>'.join(['%s: %s' % (ttc, df.iloc[row][ttc]) for ttc in tooltip_cols]) 
                                 for row in numpy.nonzero(grp_vals)[0]]
        line_dict = dict_merge(line_dict, series_dict)

        marker_dict = line_dict['marker']
        if grp in group_colours:
            marker_dict['color'] = group_colours[grp]
            
        if color_col is not None:
            marker_dict['color'] = df[color_col].values
            marker_dict['colorbar'] = {'title': color_col} #' '.join([color_col, field_caption]), 'ticksuffix': ticksuffix}

        if size_col is not None:
            marker_dict['size'] = df[size_col].to_list()
                                   
        if x_order:
            indexes = [x_order.index(x) for x in line_dict['x']]
            line_dict['x'] = [v for (i,v) in sorted(zip(indexes, line_dict['x']))]
            line_dict['y'] = [v for (i,v) in sorted(zip(indexes, line_dict['y']))]
            if 'text' in line_dict:
                line_dict['text'] = [v for (i,v) in sorted(zip(indexes, line_dict['text']))]
        pl_data.append(line_dict)          
        
    if groups_col is not None:
        for grp in sorted_groups:
            if grp in groups_available:
                grp_vals = df[groups_col] == grp
                _process_group(grp, grp_vals)
    else:
        _process_group(grp = 'Values', 
                       grp_vals = numpy.repeat(True, len(df)))
    
    data_layout = {'data': pl_data, 'layout': layout}

    return _PlotlyChartBundle(data_layout, 
                              width = width, 
                              height = height, 
                              config = default_config)

def shaded_scatter(values_df, x_item, y_item, color_item = None, title = '', value_format = '%.2f', field_caption = '', ticksuffix = '', width = 700, height=500):
    raise DeprecationWarning('shaded_scatter is deprecated and scatter with "color_col" can be used directly instead')
    
    hover_text = []
    item_format = '<br>%s' + (' ' + field_caption if field_caption else '') + ': ' + value_format
    scaling = {'%': 100}.get(ticksuffix, 1)
    for index, row in values_df.iterrows():
        t = ('Simulation: %d' % (index + 1))

        for col in values_df.columns:
            t += (item_format % (col, row[col] * scaling))
        hover_text.append(t)

    series_dict = dict(text = hover_text,
                       hoverinfo = 'text')

    if color_item is not None:
        series_dict['marker'] = {'color': [val * scaling if val > 0 else None for val in values_df[color_item]],
                                 'colorbar': {'title': ' '.join([color_item, field_caption]), 'ticksuffix': ticksuffix}
                                }
    return plotly_scatter_plot(values_df, x_col = x_item, y_col = y_item,
                        series_dict = series_dict,
                        layout = {'width': width, 
                                  'height': height,
                                  'xaxis': {'title': ' '.join([x_item, field_caption]), 'ticksuffix': ticksuffix},
                                  'yaxis': {'title': ' '.join([y_item, field_caption]), 'ticksuffix': ticksuffix},
                                  'title': title,
                                  'hovermode': 'closest'
                                 }, image_width=width, image_height=height)

def chart(dataframe, layout = dict(), column_settings = dict(), all_columns_settings = dict(), x_and_y = True, 
                     width = 800, height = 500, text_dataframe = dict(), custom_chart_data = [], col_level_separator = ': '):
    chart_data = []
    index = dataframe.index
    
    def process_column(colname, vals):
        cleaned_colname = col_level_separator.join([str(v) for v in colname]) if isinstance(colname, tuple) else colname
        coldata = {
            "name": cleaned_colname
        }
        na_mask = ~pandas.isnull(vals)
        data = vals[na_mask].values
        if x_and_y:
            coldata['x'] = index[na_mask].values
            coldata['y'] = data
        else:
            coldata['x'] = data
        
        coldata.update(all_columns_settings)

        if colname in column_settings:
            coldata.update(column_settings[colname])
                        
        if colname in text_dataframe and 'text' not in coldata:
            coldata['text'] = text_dataframe.loc[na_mask, colname].values
        chart_data.append(coldata)
       
    if isinstance(dataframe, pandas.DataFrame):
        for colname, vals in dataframe.iteritems():
            process_column(colname, vals)
    elif isinstance(dataframe, pandas.Series):
        process_column('Values', dataframe)

    layout = reduce(dict_merge, [{}, default_layout, layout])  #overwrite default_layout with given layout

    layout['width'] = width
    layout['height'] = height
        
    data_layout = {'data': chart_data + custom_chart_data, 'layout': layout}

    return _PlotlyChartBundle(data_layout, 
                              width = width, 
                              height = height, 
                              config = default_config)


