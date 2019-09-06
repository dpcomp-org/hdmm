export const domainInput = `
<form class="form-inline">
  <div class="form-group">
    <label for="inputDomainName" class="mb-2 mr-sm-2">Attribute Name </label>
    <input type="text" class="form-control mb-2 mr-sm-2" id="inputDomainName">
  </div>
  <div class="form-group">
    <label for="inputdomainType" class="mb-2 mr-sm-2">Attribute Type </label>
    <div class="input-group mb-2 mr-sm-2">
      <select class="selectpicker show-tick" id="inputdomainType" name="inputdomainType">
        <option value="Numerical" selected>Numerical</option>
        <option value="Categorical">Categorical</option>
      </select>
    </div>
  </div>
  <button type="button" class="btn btn-primary mb-2" id="callDomainModal">Next</button>
</form>
`

export const uploadFail = `
<div class="alert alert-danger alert-dismissible fade show" role="alert">
  Fail at upload, please check your file. Error message: {{error-msg}}
  <button type="button" class="close" data-dismiss="alert" aria-label="Close">
    <span aria-hidden="true">&times;</span>
  </button>
</div>
`

export const domainEditorNumerical = `
<form id={{form-id}}>
  <div class="form-row">
    <div class="col-md-3">
        <div class="form-group">
            <label for="type">Type</label>
            <select class="selectpicker show-tick" id="domainType" name="domainType">
              <option value="Numerical" selected>Numerical</option>
              <option value="Categorical">Categorical</option>
            </select>
        </div>
    </div>
    <div class="col-md-3">
        <div class="form-group">
            <label for="domainMin">Domain lower bound</label>
            <input class="form-control" type="number" name="lowerBound" id="lowerBound" value={{lower-bound}} required>
        </div>
    </div>
    <div class="col-md-3">
        <div class="form-group">
            <label for="domainMax">Domain upper bound</label>
            <input class="form-control" type="number" name="upperBound" id="upperBound" value={{upper-bound}} required>
        </div>
    </div>
    <div class="col-md-3">
        <div class="form-group">
            <label for="bucketSize">Domain bucket size</label>
            <input class="form-control" type="number" min="0" step="1" name="bucketSize" id="bucketSize" value={{bucket-size}} required>
      </div>
    </div>
  </div>
  <div class="form-row">
    <div class="col-12 d-flex">
        <button id="addDomain" type="button" class="btn btn-primary btn-sm ml-auto">Add Domain</button>
    </div>
  </div>
</form>
`

export const domainEditorCategorical = `
<form id={{form-id}}>
  <div class="form-row">
    <div class="col-md-3">
      <div class="form-group">
          <label for="type">Domain type</label>
          <select class="selectpicker show-tick" id="domainType" name="domainType">
            <option value="Numerical">Numerical</option>
            <option value="Categorical" selected>Categorical</option>
          </select>
      </div>
    </div>
    <div class="col-md-6">
      <div class="form-group">
          <label for="valueList">Domain list of values (comma separated)</label>
          <input class="form-control" type="text" name="valueList" id="valueList" value="{{distinct-value}}">
      </div>
    </div>
    <div class="col-md-2">
      <div class="form-check" style="padding-left: 2.25rem; padding-top: 2.45rem;">
        <input class="form-check-input" type="checkbox" value="" id="isOrdered">
        <label class="form-check-label" for="isOrdered">
          Is ordered?
        </label>
      </div>
    </div>
  </div>
  <div class="form-row">
    <div class="col-12 d-flex">
        <button id="addDomain" type="button" class="btn btn-primary btn-sm ml-auto">Add Domain</button>
    </div>
  </div>
</form>
`

export const domainSubmit = `
  <div class="form-row">
    <div class="col-md-11">
        <div class="form-group">
          <input type="text" id="selectedDomains" class="form-control" value=""/>
        </div>
    </div>
    <div class="col-md-1">
        <div class="form-group">
          <button id="domainSubmitButton" type="button" class="btn btn-primary btn-sm ml-auto">Submit</button>
        </div>
    </div>
 </div>
`

export const domainPicker = `<option value="{{domain-name}}">{{domain-name}}</option>`

export const domainCheckbox = `
<div class="col" style="text-align:center;">
  <div class="form-check form-check-inline">
    <input class="form-check-input" type="checkbox" id="domainCheckbox-{{domain-name}}" value="{{domain-name}}">
    <label class="form-check-label" for="domainCheckbox">{{domain-name}}</label>
  </div>
</div>
`

export const buildingBlockEditor= `
<form id="buildingBlock{{building-block-number}}">
  <div class="form-row">
    <div class="col-md-2">
      <label class="col-form-label">Building Block {{building-block-number}}</label>
    </div>
    <div class="col-md-3">
      <select class="selectpicker show-tick" id="blockDomain" name="blockDomain" required>
        <option disabled selected value="">select a domain</option>
        {{domain-options}}
      </select>
    </div>
    <div class="col-md-3">
      <select disabled class="selectpicker show-tick" id="blockType" name="blockType" required>
        <option disabled selected value="">select a building block</option>
        <option value="identity">Identity (I)</option>
        <option value="prefix">Prefix (P)</option>
        <option value="allrange">AllRange (R)</option>
        <option value="total">Total (T)</option>
        <option value="customize">Customize (C)</option>
      </select>
    </div>
    <label for="blockSummary" class="col-md-1 col-form-label">Notation</label>
    <div class="col-md-3">
      <input class="form-control" type="text" name="blockSummary" id="blockSummary" readonly>
    </div>
  </div>
</form>
`
export const kroneckerDiv = `
<div class="row kroneckerdiv" style="margin-top=15px; margin-bottom=15px;">
  <div class="col-md-4">
    <font size="6">&otimes;</font>
  </div>
</div>
`

export const workloadRow = `
<div class="form-row" id="workloadRow{{workloadRow-number}}">
  <label for="selectedWorkload" class="col-md-2 col-form-label" style="text-align: right;">Workload {{workloadRow-number}}</label>
  <div class="col-md-6">
      <div class="form-group">
        <input type="text" id="selectedWorkload" class="form-control" value="{{workloadRow-string}}" readonly />
      </div>
  </div>
  <label for="workloadWeight" class="col-md-1 col-form-label" style="text-align: right;">Weight</label>
  <div class="col-md-2">
      <div class="form-group">
        <input type="number" id="workloadWeight" class="form-control" value="{{workloadRow-weight}}" min="0" step="0.1"/>
      </div>
  </div>
  <div class="col-md-1" style="padding-top: 5px;">
    <a href="#" id="removeWorkload"><i class="fas fa-times-circle" style="font-size:20px; color:#dc3545;"></i></a>
  </div>
</div>
`
export const workloadOversize = `
<div class="alert alert-danger alert-dismissible fade show" role="alert">
  Workloads oversize, please try smaller workloads. Error message: Server out of memory.
  <button type="button" class="close" data-dismiss="alert" aria-label="Close">
    <span aria-hidden="true">&times;</span>
  </button>
</div>
`

export const customizeInputError = `
<div class="alert alert-danger alert-dismissible fade show" role="alert">
  {{error-msg}}
  <button type="button" class="close" data-dismiss="alert" aria-label="Close">
    <span aria-hidden="true">&times;</span>
  </button>
</div>
`

export const multiWorkloadsHDMMTable = `
<table class="table table-bordered" style="width:80%">
<caption> Table1: HDMM1 - Evalute each workload separately </caption>
<thead>
  <tr>
    <th width="15%" style="vertical-align : middle;text-align:center;">Workload ID</th>
    <th width="45%" style="vertical-align : middle;text-align:center;">Workload</th>
    <th width="20%" style="vertical-align : middle;text-align:center;">Number of Queries</th>
    <th width="20%" style="vertical-align : middle;text-align:center;">Expected Root MSE</th>
  </tr>
</thead>
<tbody>
  {{summaryRows}}
</tbody>
</table>
`
export const multiWorkloadsSummaryTable = `
<table class="table table-bordered" style="width:80%">
<caption>Table2: Compare different methods for multiple workloads</caption>
<thead>
  <tr>
    <th style="vertical-align : middle;text-align:center;">Method</th>
    <th style="vertical-align : middle;text-align:center;">Description</th>
    <th style="vertical-align : middle;text-align:center;">Expected Root MSE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="vertical-align : middle;text-align:center;">Identity</td>
    <td>Identity with Laplace noise</td>
    <td style="vertical-align : middle;text-align:center;">{{ideError}}</td>
  </tr>
  <tr>
    <td style="vertical-align : middle;text-align:center;">HDMM</td>
    <td>Best HDMM Strategy</td>
    <td style="vertical-align : middle;text-align:center;">{{hdmmError}}</td>
  </tr>
</tbody>
</table>
`
export const singleWorkloadSummaryTable = `
<table class="table table-bordered" style="width:80%">
<caption>Table1: Compare different methods for single workload</caption>
<thead>
  <tr>
    <th width="38%" style="vertical-align : middle;text-align:center;">Workload</th>
    <th width="20%" style="vertical-align : middle;text-align:center;">Number of Queries</th>
    <th width="30%" style="vertical-align : middle;text-align:center;">Method &amp; Description</th>
    <th width="12%" style="vertical-align : middle;text-align:center;">Expected Root MSE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2" style="vertical-align : middle;text-align:left;">{{workloadString}}</td>
    <td rowspan="2" style="vertical-align : middle;text-align:center;">{{queryNumber}}</td>
    <td>Identity with Laplace noise (Identity)</td>
    <td style="vertical-align : middle;text-align:center;">{{ideError}}</td>
  </tr>
  <tr>
    <td>High-dimensional Matrix Mechanism (HDMM)</td>
    <td style="vertical-align : middle;text-align:center;">{{hdmmError}}</td>
  </tr>
</tbody>
</table>
`

export const domainMetaTable = `
<table class="table table-bordered" style="width:100%">
<caption>Table0: Available domains for creating workloads.</caption>
<thead>
  <tr>
    <th width="30%">Name</th>
    <th width="70%">Configurations</th>
  </tr>
</thead>
<tbody>
  {{domain-configs}}
</tbody>
</table>
`
